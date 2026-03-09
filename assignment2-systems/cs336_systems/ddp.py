from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn


def _iter_unique_parameters(
    module: nn.Module,
    *,
    requires_grad_only: bool,
) -> list[nn.Parameter]:    # 返回去重后的参数列表，兼容 tied weights。
    params: list[nn.Parameter] = []
    seen_param_ids: set[int] = set()

    for param in module.parameters():
        if requires_grad_only and not param.requires_grad:
            continue
        # tied weights 会在 parameters() 里重复出现，这里按对象 id 去重。
        if id(param) in seen_param_ids:
            continue
        seen_param_ids.add(id(param))
        params.append(param)

    return params


def _broadcast_module_state(module: nn.Module) -> None:    # 用 rank 0 的参数/缓冲区覆盖所有其他 rank。
    if not dist.is_available() or not dist.is_initialized():
        return

    with torch.no_grad():
        # state_dict() 同时覆盖参数和 buffer，初始化后各 rank 状态应完全一致。
        for tensor in module.state_dict().values():
            dist.broadcast(tensor, src=0)


def _average_gradient(grad: torch.Tensor, world_size: int) -> None:    # 对单个梯度做 all-reduce 并取平均。
    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    grad.div_(world_size)


def _average_bucket_gradients(
    bucket: Iterable[nn.Parameter],
    world_size: int,
) -> None:    # 把一个 bucket 的梯度展平通信，再拷回各参数。
    grads = [param.grad for param in bucket if param.grad is not None]
    if not grads:
        return

    # 先展平成一个连续大张量，再做一次 collective。
    flat_grad = torch.cat([grad.reshape(-1) for grad in grads])
    dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
    flat_grad.div_(world_size)

    offset = 0
    for grad in grads:
        numel = grad.numel()
        grad.copy_(flat_grad[offset : offset + numel].view_as(grad))
        offset += numel


def _build_buckets(
    params: list[nn.Parameter],
    bucket_size_mb: float | None,
) -> list[list[nn.Parameter]]:    # 按参数大小把参数切成若干 bucket。
    if bucket_size_mb is None:
        return [params]

    bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
    if bucket_size_bytes <= 0:
        return [[param] for param in params]

    buckets: list[list[nn.Parameter]] = []
    current_bucket: list[nn.Parameter] = []
    current_bucket_size = 0

    for param in params:
        param_size = param.numel() * param.element_size()
        # 超过阈值就切到下一个 bucket，保持每个 bucket 大小可控。
        if current_bucket and current_bucket_size + param_size > bucket_size_bytes:
            buckets.append(current_bucket)
            current_bucket = []
            current_bucket_size = 0

        current_bucket.append(param)
        current_bucket_size += param_size

    if current_bucket:
        buckets.append(current_bucket)

    return buckets


class _BaseDDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
    ) -> None:    # 保存底层模块，并在初始化时把模型状态从 rank 0 广播出去。
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        _broadcast_module_state(self.module)

    def forward(self, *args, **kwargs):    # 直接把 forward 委托给底层模块。
        return self.module(*args, **kwargs)


class DDP(_BaseDDP):
    def __init__(
        self,
        module: nn.Module,
    ) -> None:    # 逐参数同步梯度的 DDP 包装器。
        super().__init__(module)
        self.grad_params = _iter_unique_parameters(
            self.module,
            requires_grad_only=True,
        )
        # 记录每个参数当前尚未 wait 的异步 all-reduce work。
        self.pending_works: dict[int, dist.Work] = {}
        # 保存 hook handle，避免 hook 因引用丢失而被移除。
        self.hook_handles = []

        for param in self.grad_params:
            handle = param.register_post_accumulate_grad_hook(
                self._make_post_accumulate_hook(param)
            )
            self.hook_handles.append(handle)

    def _make_post_accumulate_hook(
        self,
        param: nn.Parameter,
    ):    # 为单个参数创建“梯度一就绪就异步 all-reduce”的 hook。
        param_id = id(param)

        def hook(grad_param: nn.Parameter) -> None:
            if self.world_size == 1 or grad_param.grad is None:
                return
            # 同一轮 backward 中，一个参数只发起一次异步通信。
            if param_id in self.pending_works:
                return
            self.pending_works[param_id] = dist.all_reduce(
                grad_param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )

        return hook

    def finish_gradient_synchronization(self) -> None:    # backward 后等待所有异步 all-reduce 完成，并把梯度取平均。
        if self.world_size == 1:
            return

        for param in self.grad_params:
            if param.grad is None:
                continue
            work = self.pending_works.pop(id(param), None)
            # 在 optimizer.step() 前把所有异步通信真正收完。
            if work is not None:
                work.wait()
            param.grad.div_(self.world_size)

class DDPBucketed(_BaseDDP):
    def __init__(
        self,
        module: nn.Module,
        bucket_size_mb: float | None,
    ) -> None:    # 按 bucket 同步梯度的 DDP 包装器。
        super().__init__(module)
        grad_params = _iter_unique_parameters(
            self.module,
            requires_grad_only=True,
        )
        self.buckets = _build_buckets(
            grad_params,
            bucket_size_mb,
        )

    def start_train_batch(self) -> None:    # 当前朴素实现没有异步状态，这里保持空操作。
        return None

    def finish_gradient_synchronization(self) -> None:    # backward 后按 bucket 做 all-reduce。
        if self.world_size == 1:
            return

        for bucket in self.buckets:
            _average_bucket_gradients(bucket, self.world_size)


class DDPFlatGradients(_BaseDDP):
    def __init__(
        self,
        module: nn.Module,
    ) -> None:    # 把所有梯度拼成一个扁平张量来同步的 DDP 包装器。
        super().__init__(module)
        self.grad_params = _iter_unique_parameters(
            self.module,
            requires_grad_only=True,
        )

    def finish_gradient_synchronization(self) -> None:    # backward 后只做一次扁平 all-reduce。
        if self.world_size == 1:
            return

        _average_bucket_gradients(self.grad_params, self.world_size)
