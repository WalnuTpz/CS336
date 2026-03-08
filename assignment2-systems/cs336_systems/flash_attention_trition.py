import math
from typing import Type

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    raise NotImplementedError


class FlashAttentionForwardTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Tensor,          # (..., N, D)
        K: Tensor,          # (..., N, D)
        V: Tensor,          # (..., N, D)
        is_causal: bool = False,
    ) -> Tensor:           # (..., N, D)
        raise NotImplementedError

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError


def get_flash_autograd_function_triton() -> Type:
    return FlashAttentionForwardTriton