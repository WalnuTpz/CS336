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
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_start = query_tile_index * Q_TILE_SIZE    # 当前 q_tile 的起始 query 行号
    offs_q = q_start + tl.arange(0, Q_TILE_SIZE)    # 当前 q_tile 中每一行的全局偏移量
    valid_q = offs_q < N_QUERIES    # 合法的 query 行的位置

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(    # 把 K 视为 (D, N_KEYS) 而不是 (N_KEYS, D)，方便后续 S = Q @ K^T 的计算
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        offs_k = k_start + tl.arange(0, K_TILE_SIZE)    # 当前 k_tile 中每一行的全局偏移量
        valid_k = offs_k < N_KEYS    # 合法的 key 行的位置

        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # 分块 softmax
        s_ij = tl.dot(q, k) * scale
        valid = valid_q[:, None] & valid_k[None, :]  # (Q_TILE_SIZE, K_TILE_SIZE)
        if IS_CAUSAL:
            valid = valid & (offs_q[:, None] >= offs_k[None, :])
        s_ij = tl.where(valid, s_ij, float("-inf"))

        # 计算并更新最大值
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        safe_m_i = tl.where(valid_q, m_i, 0.0)    # 将非法位置的值改为 0，防止后续计算时出现 nan，后面也同理
        safe_m_new = tl.where(valid_q, m_new, 0.0)

        # 计算新的输出块
        alpha = tl.where(valid_q, tl.exp(safe_m_i - safe_m_new), 0.0)
        p_tilde = tl.where(valid, tl.exp(s_ij - safe_m_new[:, None]),0.0)
        l_new = alpha * l_i + tl.sum(p_tilde, axis=1)
        o_i = alpha[:, None] * o_i
        o_i = tl.dot(p_tilde.to(v.dtype), v, acc=o_i)    # o_i = o_i + p_tilde @ v

        m_i = m_new
        l_i = l_new
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    o_i = o_i / l_i[:, None]

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