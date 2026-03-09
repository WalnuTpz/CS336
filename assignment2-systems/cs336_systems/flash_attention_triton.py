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

    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    L_ptrs = L_ptr + batch_index * stride_lb + offs_q * stride_lq
    L_vals = m_i + tl.log(l_i)
    tl.store(L_ptrs, L_vals, mask=valid_q)


@triton.jit
def flash_bwd_prep_kernel(
    dO_ptr, O_ptr, D_vec_ptr,
    stride_dob, stride_doq, stride_dod,
    stride_ob, stride_oq, stride_od,
    stride_db, stride_dq,
    N_QUERIES,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_start = query_tile_index * Q_TILE_SIZE
    offs_q = q_start + tl.arange(0, Q_TILE_SIZE)
    valid_q = offs_q < N_QUERIES

    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
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

    dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    O = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    # 预处理 D 向量：D_i = sum_d dO_i[d] * O_i[d]
    D_vec = tl.sum(dO * O, axis=1)
    D_vec_ptrs = D_vec_ptr + batch_index * stride_db + offs_q * stride_dq
    tl.store(D_vec_ptrs, D_vec, mask=valid_q)


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_vec_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    k_start = key_tile_index * K_TILE_SIZE
    offs_k = k_start + tl.arange(0, K_TILE_SIZE)
    valid_k = offs_k < N_KEYS

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    dk_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dv_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    # 固定当前 k_tile，遍历所有 q_tile 来累计 dK_j 和 dV_j
    for q_start in range(0, N_QUERIES, Q_TILE_SIZE):
        offs_q = q_start + tl.arange(0, Q_TILE_SIZE)
        valid_q = offs_q < N_QUERIES

        Q_block_ptr = tl.make_block_ptr(
            base=Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        dO_block_ptr = tl.make_block_ptr(
            base=dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        L_ptrs = L_ptr + batch_index * stride_lb + offs_q * stride_lq
        l = tl.load(L_ptrs, mask=valid_q, other=0.0)
        D_vec_ptrs = D_vec_ptr + batch_index * stride_db + offs_q * stride_dq
        d_i = tl.load(D_vec_ptrs, mask=valid_q, other=0.0)

        # S_i^(j) = Q_i @ K_j^T / sqrt(d)
        s_ij = tl.dot(q, tl.trans(k)) * scale
        valid = valid_q[:, None] & valid_k[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_q[:, None] >= offs_k[None, :])
        s_ij = tl.where(valid, s_ij, float("-inf"))

        # P_i^(j) = exp(S_i^(j) - L_i)，dP_i^(j) = dO_i @ V_j^T
        p = tl.where(valid, tl.exp(s_ij - l[:, None]), 0.0)
        dP = tl.dot(dO, tl.trans(v))
        # dS_i^(j) = P_i^(j) * (dP_i^(j) - D_i)，1 / sqrt(d) 放到最后统一乘
        dS = tl.where(valid, p * (dP - d_i[:, None]), 0.0)

        # dV_j += (P_i^(j))^T @ dO_i，dK_j += (dS_i^(j))^T @ Q_i
        dv_j = tl.dot(tl.trans(p), dO, acc=dv_j)
        dk_j = tl.dot(tl.trans(dS), q, acc=dk_j)

    tl.store(dK_block_ptr, (scale * dk_j).to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dv_j.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_vec_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_start = query_tile_index * Q_TILE_SIZE
    offs_q = q_start + tl.arange(0, Q_TILE_SIZE)
    valid_q = offs_q < N_QUERIES

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    L_ptrs = L_ptr + batch_index * stride_lb + offs_q * stride_lq
    l = tl.load(L_ptrs, mask=valid_q, other=0.0)
    D_vec_ptrs = D_vec_ptr + batch_index * stride_db + offs_q * stride_dq
    d_i = tl.load(D_vec_ptrs, mask=valid_q, other=0.0)

    dq_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # 固定当前 q_tile，遍历所有 k_tile 来累计 dQ_i
    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        offs_k = k_start + tl.arange(0, K_TILE_SIZE)
        valid_k = offs_k < N_KEYS

        K_block_ptr = tl.make_block_ptr(
            base=K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            base=V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        # S_i^(j) = Q_i @ K_j^T / sqrt(d)
        s_ij = tl.dot(q, tl.trans(k)) * scale
        valid = valid_q[:, None] & valid_k[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_q[:, None] >= offs_k[None, :])
        s_ij = tl.where(valid, s_ij, float("-inf"))

        # P_i^(j) = exp(S_i^(j) - L_i)，dP_i^(j) = dO_i @ V_j^T
        p = tl.where(valid, tl.exp(s_ij - l[:, None]), 0.0)
        dP = tl.dot(dO, tl.trans(v))
        # dS_i^(j) = P_i^(j) * (dP_i^(j) - D_i)，1 / sqrt(d) 放到最后统一乘
        dS = tl.where(valid, p * (dP - d_i[:, None]), 0.0)
        # dQ_i += dS_i^(j) @ K_j
        dq_i = tl.dot(dS, k, acc=dq_i)

    tl.store(dQ_block_ptr, (scale * dq_i).to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))


class FlashAttentionForwardTriton(torch.autograd.Function):
    def forward(
        ctx,
        Q: Tensor,  # (..., N, D)
        K: Tensor,  # (..., N, D)
        V: Tensor,  # (..., N, D)
        is_causal: bool = False,
    ) -> Tensor:  # (..., N, D)
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Triton kernel requires CUDA tensors"
        assert Q.shape == K.shape == V.shape
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), \
            "Please pass contiguous Q/K/V"
        assert Q.dtype in (torch.float16, torch.bfloat16, torch.float32)

        *batch_dims, N, D = Q.shape
        B = math.prod(batch_dims) if len(batch_dims) > 0 else 1

        # 将前导维度展平
        Q_ = Q.reshape(B, N, D)
        K_ = K.reshape(B, N, D)
        V_ = V.reshape(B, N, D)

        O_ = torch.empty_like(Q_)
        L_ = torch.empty((B, N), device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        scale = 1.0 / math.sqrt(D)

        grid = (triton.cdiv(N, Q_TILE_SIZE), B)

        flash_fwd_kernel[grid](
            Q_, K_, V_,
            O_, L_,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            O_.stride(0), O_.stride(1), O_.stride(2),
            L_.stride(0), L_.stride(1),
            N, N,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=is_causal,
            num_warps=4,
            num_stages=2,
        )

        O = O_.reshape(*batch_dims, N, D)
        L = L_.reshape(*batch_dims, N)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        assert Q.is_cuda and K.is_cuda and V.is_cuda and O.is_cuda and dO.is_cuda and L.is_cuda
        assert Q.shape == K.shape == V.shape == O.shape == dO.shape

        *batch_dims, N, D = Q.shape
        B = math.prod(batch_dims) if len(batch_dims) > 0 else 1

        # backward 中的 dO 不保证是 contiguous 的，这里手动整理一下
        dO = dO.contiguous()

        # 将前导维度展平
        Q_ = Q.reshape(B, N, D)
        K_ = K.reshape(B, N, D)
        V_ = V.reshape(B, N, D)
        O_ = O.reshape(B, N, D)
        dO_ = dO.reshape(B, N, D)
        L_ = L.reshape(B, N)

        D_vec_ = torch.empty((B, N), device=Q.device, dtype=torch.float32)
        dQ_ = torch.empty_like(Q_)
        dK_ = torch.empty_like(K_)
        dV_ = torch.empty_like(V_)

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        scale = 1.0 / math.sqrt(D)

        # 先计算 backward 会反复用到的 D 向量
        prep_grid = (triton.cdiv(N, Q_TILE_SIZE), B)
        flash_bwd_prep_kernel[prep_grid](
            dO_, O_, D_vec_,
            dO_.stride(0), dO_.stride(1), dO_.stride(2),
            O_.stride(0), O_.stride(1), O_.stride(2),
            D_vec_.stride(0), D_vec_.stride(1),
            N,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            num_warps=4,
            num_stages=2,
        )

        # 按 k_tile 累计 dK 和 dV
        dkdv_grid = (triton.cdiv(N, K_TILE_SIZE), B)
        flash_bwd_dkdv_kernel[dkdv_grid](
            Q_, K_, V_,
            dO_, L_, D_vec_,
            dK_, dV_,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            dO_.stride(0), dO_.stride(1), dO_.stride(2),
            L_.stride(0), L_.stride(1),
            D_vec_.stride(0), D_vec_.stride(1),
            dK_.stride(0), dK_.stride(1), dK_.stride(2),
            dV_.stride(0), dV_.stride(1), dV_.stride(2),
            N, N,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=ctx.is_causal,
            num_warps=4,
            num_stages=2,
        )

        # 按 q_tile 累计 dQ
        dq_grid = (triton.cdiv(N, Q_TILE_SIZE), B)
        flash_bwd_dq_kernel[dq_grid](
            Q_, K_, V_,
            dO_, L_, D_vec_,
            dQ_,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            dO_.stride(0), dO_.stride(1), dO_.stride(2),
            L_.stride(0), L_.stride(1),
            D_vec_.stride(0), D_vec_.stride(1),
            dQ_.stride(0), dQ_.stride(1), dQ_.stride(2),
            N, N,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=ctx.is_causal,
            num_warps=4,
            num_stages=2,
        )
        dQ = dQ_.reshape(*batch_dims, N, D)
        dK = dK_.reshape(*batch_dims, N, D)
        dV = dV_.reshape(*batch_dims, N, D)
        return dQ, dK, dV, None


def get_flash_autograd_function_triton() -> Type:
    return FlashAttentionForwardTriton
