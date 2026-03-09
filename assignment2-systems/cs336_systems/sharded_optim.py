from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist


def _iter_unique_params_from_groups(
    param_groups: list[dict[str, Any]],
) -> list[torch.nn.Parameter]:    # 按参数组顺序收集去重后的参数列表，兼容 tied weights。
    params: list[torch.nn.Parameter] = []
    seen_param_ids: set[int] = set()

    for group in param_groups:
        for param in group["params"]:
            if id(param) in seen_param_ids:
                continue
            seen_param_ids.add(id(param))
            params.append(param)

    return params


def _build_local_param_groups(
    global_param_groups: list[dict[str, Any]],
    param_owner_map: dict[int, int],
    rank: int,
) -> list[dict[str, Any]]:    # 从全局参数组中过滤出当前 rank 负责更新的那一部分参数。
    local_param_groups: list[dict[str, Any]] = []
    seen_param_ids: set[int] = set()

    for group in global_param_groups:
        local_params: list[torch.nn.Parameter] = []
        for param in group["params"]:
            param_id = id(param)
            if param_id in seen_param_ids:
                continue
            if param_owner_map[param_id] != rank:
                continue
            seen_param_ids.add(param_id)
            local_params.append(param)

        if not local_params:
            continue

        local_group = {
            key: value
            for key, value in group.items()
            if key != "params"
        }
        local_group["params"] = local_params
        local_param_groups.append(local_group)

    return local_param_groups


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs,
    ) -> None:    # 只在当前 rank 上保存一部分优化器状态，step 后再同步更新后的参数。
        params = list(params)
        super().__init__(params, kwargs)

        self.optimizer_cls = optimizer_cls
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.global_params = _iter_unique_params_from_groups(self.param_groups)
        self.param_owner_map = {
            id(param): idx % self.world_size
            for idx, param in enumerate(self.global_params)
        }
        self.local_param_groups = _build_local_param_groups(
            self.param_groups,
            self.param_owner_map,
            self.rank,
        )
        self.local_optimizer = (
            self.optimizer_cls(self.local_param_groups)
            if self.local_param_groups
            else None
        )
        if self.local_optimizer is not None:
            # 对外暴露的 state 只包含当前 rank 持有的那部分优化器状态。
            self.state = self.local_optimizer.state

        self._broadcast_parameters(src=0)

    @torch.no_grad()
    def _broadcast_parameters(
        self,
        src: int,
    ) -> None:    # 用指定 rank 的参数覆盖所有其他 rank，确保各副本权重一致。
        if self.world_size == 1:
            return

        for param in self.global_params:
            dist.broadcast(param.data, src=src)

    def zero_grad(
        self,
        set_to_none: bool = True,
    ) -> None:    # 清空所有参数的梯度，而不只是本 rank 负责更新的那一部分。
        seen_param_ids: set[int] = set()

        for group in self.param_groups:
            for param in group["params"]:
                param_id = id(param)
                if param_id in seen_param_ids:
                    continue
                seen_param_ids.add(param_id)

                if param.grad is None:
                    continue
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.zero_()

    @torch.no_grad()
    def step(
        self,
        closure=None,
    ):    # 当前 rank 只更新自己拥有的参数，然后把更新结果广播给所有 rank。
        loss = None
        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.world_size == 1:
            return loss

        for param in self.global_params:
            owner_rank = self.param_owner_map[id(param)]
            dist.broadcast(param.data, src=owner_rank)

        return loss
