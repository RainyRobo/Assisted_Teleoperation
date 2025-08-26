# -*- coding: utf-8 -*-
from __future__ import annotations
import jax, jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from typing import Optional, Tuple

# -------------------------
# 位置编码 / RoPE / 杂项
# -------------------------
def posemb_sincos_any(pos_idx: jnp.ndarray, emb_dim: int,
                      min_period: float = 1.0, max_period: float = 1e4) -> jnp.ndarray:
    assert emb_dim % 2 == 0, "embedding_dim 必须为偶数"
    pos = pos_idx.astype(jnp.float32)[..., None]              # (..., 1)
    i   = jnp.arange(emb_dim // 2, dtype=jnp.float32)[None, :]# (1, d/2)
    frac   = i / jnp.maximum(emb_dim // 2 - 1, 1)
    period = min_period * (max_period / min_period) ** frac   # (1, d/2)
    ang    = (2 * jnp.pi / period) * pos                      # (..., d/2)
    pe     = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    return pe

def build_rope_cache(T: int, Dh: int, base: float = 10_000.0, dtype=jnp.float32):
    assert Dh % 2 == 0, "head_dim (Dh) must be even for RoPE."
    half = Dh // 2
    freqs = 1.0 / (base ** (jnp.arange(0, half, dtype=dtype) / half))
    pos = jnp.arange(T, dtype=dtype)[:, None]                 # [T,1]
    angles = pos * freqs[None, :]                             # [T,half]
    cos = jnp.repeat(jnp.cos(angles), 2, axis=-1)             # [T,Dh]
    sin = jnp.repeat(jnp.sin(angles), 2, axis=-1)             # [T,Dh]
    return cos, sin

def rope_apply(x_hd: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    Dh = x_hd.shape[-1]
    assert Dh % 2 == 0
    x2 = x_hd.reshape(*x_hd.shape[:-1], Dh // 2, 2)
    x1, x2_im = x2[..., 0], x2[..., 1]
    y1 = x1 * cos[..., :Dh//2] - x2_im * sin[..., :Dh//2]
    y2 = x1 * sin[..., :Dh//2] + x2_im * cos[..., :Dh//2]
    y = jnp.stack([y1, y2], axis=-1).reshape(*x_hd.shape)
    return y

def rope_cos_sin_from_pos(pos_idx: jnp.ndarray, Dh: int, base: float = 10_000.0, dtype=jnp.float32):
    """给定绝对位置索引（可按样本不同），生成 RoPE 的 cos/sin。
    pos_idx: [B, T] 位置（绝对整数），Dh 必须为偶数
    返回: cos, sin -> [B, T, Dh]
    """
    assert Dh % 2 == 0, "RoPE需要偶数的 head_dim"
    B, T = pos_idx.shape
    half = Dh // 2
    freqs = 1.0 / (base ** (jnp.arange(0, half, dtype=dtype) / half))  # [half]
    ang = pos_idx.astype(dtype)[..., None] * freqs[None, None, :]      # [B,T,half]
    cos = jnp.repeat(jnp.cos(ang), 2, axis=-1)  # [B,T,Dh]
    sin = jnp.repeat(jnp.sin(ang), 2, axis=-1)  # [B,T,Dh]
    return cos, sin


def masked_mean(x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, axis=None, eps: float = 1e-8):
    if mask is None:
        return jnp.mean(x, axis=axis)
    m = mask.astype(x.dtype)
    if axis is None:
        return (x * m).sum() / (m.sum() + eps)
    return (x * m).sum(axis=axis) / (m.sum(axis=axis) + eps)

def pairwise_sq_dists(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    a2 = jnp.sum(A**2, axis=-1, keepdims=True)           # [B,N,1]
    b2 = jnp.sum(B**2, axis=-1, keepdims=True).transpose(0,2,1)  # [B,1,M]
    ab = jnp.einsum('bie,bje->bij', A, B)                # [B,N,M]
    return a2 + b2 - 2.0 * ab

import jax
import jax.numpy as jnp

def get_history_window(
    his_state_data: jnp.ndarray,   # [B, Ts, D]
    his_mask: jnp.ndarray,         # [B, Ts] (bool)
    curr_id,                       # 标量 int 或 [B] int32：窗口右端(不含)位置
    window_size                    # 标量 int 或 [B] int32：窗口长度
):
    """
    取每个样本的历史窗口 [curr_id - window_size, curr_id)，
    不足则左侧“逻辑上”pad（用 mask=False 标注，无需 jnp.pad）。
    返回:
      win_data: [B, W, D]
      win_mask: [B, W] (bool)
      q_pos_idx: [B, W] （可用于 RoPE 的绝对位置；对无效位可随意/0）
    说明：
      - 使用 take_along_axis 按索引 gather，避免动态切片限制；
      - 对超界的索引使用 clip 到 [0, Ts-1]，随后用 win_mask 将这些位置置无效；
      - his_mask 会与有效时间窗口共同决定 win_mask。
    """
    B, Ts, D = his_state_data.shape

    # 统一形状
    if jnp.ndim(curr_id) == 0:
        curr_id = jnp.full((B,), curr_id, dtype=jnp.int32)
    else:
        curr_id = curr_id.astype(jnp.int32)
    if jnp.ndim(window_size) == 0:
        W = int(window_size)  # 静态窗口长度
        window_size_vec = jnp.full((B,), W, dtype=jnp.int32)
    else:
        # 若 window_size 为 [B]，取其最大值作为固定窗口长度 W（静态），其余样本用 mask 处理
        W = int(jax.device_get(jnp.max(window_size)))  # 转成 Python int
        window_size_vec = window_size.astype(jnp.int32)

    # 目标窗口的绝对时间索引（未裁剪）：base_idx[b, t] = curr_id[b] - W + t
    t_rel = jnp.arange(W, dtype=jnp.int32)[None, :]               # [1, W]
    base_idx = curr_id[:, None] - W + t_rel                       # [B, W]  可能为负或 >= Ts

    # 有效时间条件：0 <= idx < curr_id（右开区间）
    valid_time = (base_idx >= 0) & (base_idx < curr_id[:, None])  # [B, W]

    # 裁剪到合法范围用于 gather
    idx = jnp.clip(base_idx, 0, Ts - 1)                           # [B, W]

    # gather 数据与原始 mask
    idx_exp = idx[..., None].repeat(D, axis=-1)                   # [B, W, D]
    win_data = jnp.take_along_axis(his_state_data, idx_exp, axis=1)      # [B, W, D]
    win_mask_raw = jnp.take_along_axis(his_mask, idx, axis=1)            # [B, W]

    # 最终 mask：必须在时间上有效 且 原始位置非 padding
    win_mask = valid_time & win_mask_raw                                  # [B, W]

    # 对无效位的数据置零（可选，通常更安全）
    win_data = win_data * win_mask[..., None]

    # RoPE 位置索引（可选）：对无效位设 0 即可，反正有 mask
    q_pos_idx = jnp.where(win_mask, base_idx, 0)                          # [B, W]

    return win_data, win_mask

def kl_per_dim_diag_gauss(mu_q, logvar_q, mu_p, logvar_p):
    # 逐维 KL(q||p)，形状与输入相同的 [B, T, Z]
    # 写成更数值稳定的形式，避免两次 exp：
    return 0.5 * (
        jnp.exp(logvar_q - logvar_p)                               # var_q/var_p
        + jnp.exp(-logvar_p) * (mu_q - mu_p) ** 2                  # (mu diff)^2 / var_p
        - 1.0
        + (logvar_p - logvar_q)
    )

def kl_balanced_with_freebits(
    mu_z, logvar_z, mu_p, logvar_p, mask,
    alpha=0.8, free_bits=0.2, reduce="mean", eps=1e-8
):
    # 1) 两个方向
    kl_q_p = kl_per_dim_diag_gauss(mu_z, logvar_z, jax.lax.stop_gradient(mu_p), jax.lax.stop_gradient(logvar_p))
    kl_p_q = kl_per_dim_diag_gauss(jax.lax.stop_gradient(mu_z), jax.lax.stop_gradient(logvar_z), mu_p, logvar_p)
    kl_dim = alpha * kl_q_p + (1 - alpha) * kl_p_q                  # [B,T,Z]

    # 2) 先对 (B,T) 做掩码均值，得到每维的平均 KL
    m = mask.astype(kl_dim.dtype)[..., None]                         # [B,T,1]
    kl_dim_mean = (kl_dim * m).sum(axis=(0,1)) / (m.sum(axis=(0,1)) + eps)  # [Z]

    # 3) 逐维阈值（free-bits）
    thr = jnp.maximum(kl_dim_mean - free_bits, 0.0)                  # [Z]
    kl_dim_fb = jnp.maximum(kl_dim - thr[None, None, :], 0.0)        # [B,T,Z]

    # 4) 汇总
    kl_step = kl_dim_fb.sum(-1)                                      # [B,T]
    if reduce == "none":
        return kl_step
    return (kl_step * mask.astype(kl_step.dtype)).sum() / (mask.sum() + eps)


# -------------------------
# TCC（cycle-back regression）
# -------------------------
def l2_normalize(x, axis=-1, eps=1e-8):
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + eps)

def pairwise_sq_dists_mean(A, B):  # 用均值而不是求和，尺度更稳
    # A: [B, Ta, E], B: [B, Tb, E]
    diff = A[:, :, None, :] - B[:, None, :, :]
    return jnp.mean(diff**2, axis=-1)  # [B, Ta, Tb]

def tcc_cycle_back_regression(
    U, V, mask_u=None, mask_v=None,
    tau: float = 0.1, lambda_logvar: float = 1e-3,
    var_floor: float = 1e-4, eps: float = 1e-8, ent_reg: float = 0.0,
):
    # 1) 特征归一化
    U = l2_normalize(U, -1)
    V = l2_normalize(V, -1)

    # 2) U→V 注意力（带稳定化与掩码）
    d_uv = pairwise_sq_dists_mean(U, V)                      # [B, Tu, Tv]
    logits_uv = -d_uv / tau
    if mask_v is not None:
        logits_uv = jnp.where(mask_v[:, None, :], logits_uv, -jnp.inf)
    logits_uv = logits_uv - jnp.max(logits_uv, axis=-1, keepdims=True)
    alpha = jax.nn.softmax(logits_uv, axis=-1)               # [B, Tu, Tv]
    v_e = jnp.einsum('bij,bje->bie', alpha, V)               # [B, Tu, E]

    # 3) 回到 U 的注意力
    d_ue = pairwise_sq_dists_mean(v_e, U)                    # [B, Tu, Tu]
    logits_ue = -d_ue / tau
    if mask_u is not None:
        logits_ue = jnp.where(mask_u[:, None, :], logits_ue, -jnp.inf)
    logits_ue = logits_ue - jnp.max(logits_ue, axis=-1, keepdims=True)
    beta = jax.nn.softmax(logits_ue, axis=-1)                # [B, Tu, Tu]

    # 4) 时间轴标准化到 [0,1]
    B, Tu = U.shape[0], U.shape[1]
    if mask_u is None:
        valid_len = jnp.full((B, 1), Tu, dtype=U.dtype)
        m_u = jnp.ones((B, Tu), bool)
    else:
        m_u = mask_u.astype(bool)
        valid_len = jnp.sum(m_u, axis=1, keepdims=True).astype(U.dtype)
    scale = jnp.maximum(valid_len - 1.0, 1.0)                # [B,1]

    idx = (jnp.arange(Tu, dtype=U.dtype)[None, None, :] / scale[:, :, None])  # [B,1,Tu] in [0,1]
    mu  = jnp.sum(beta * idx, axis=-1)                       # [B, Tu]
    var = jnp.sum(beta * (idx - mu[..., None])**2, axis=-1)  # [B, Tu]
    var = jnp.maximum(var, var_floor)

    tgt = jnp.arange(Tu, dtype=U.dtype)[None, :] / scale     # [B, Tu]

    per_step = ((tgt - mu)**2) / var + lambda_logvar * jnp.log(var)  # [B, Tu]

    # 掩码均值
    def masked_mean(x, mask):
        if mask is None: return jnp.mean(x)
        m = mask.astype(x.dtype)
        return (x * m).sum() / (m.sum() + eps)

    loss = masked_mean(per_step, m_u)

    # 可选：β 的熵正则，避免过于尖锐
    if ent_reg > 0.0:
        ent = -jnp.sum(beta * jnp.log(beta + 1e-12), axis=-1)  # [B, Tu]
        loss = loss - ent_reg * masked_mean(ent, m_u)

    return loss


# -------------------------
# Attention 连续性正则（TV / Laplacian）
# -------------------------
def attn_tv_l1(attn: jnp.ndarray, kv_mask: Optional[jnp.ndarray]):
    # attn: [B,H,Q,N], kv_mask: [B,N] or None
    diff = attn[..., 1:] - attn[..., :-1]                      # [B,H,Q,N-1]
    if kv_mask is not None:
        m = kv_mask[:, None, None, 1:] * kv_mask[:, None, None, :-1]
        diff = diff * m
        denom = m.sum()
    else:
        denom = diff.size
    return jnp.sum(jnp.abs(diff)) / (denom + 1e-9)

def attn_laplace_l2(attn: jnp.ndarray, kv_mask: Optional[jnp.ndarray]):
    d2 = attn[..., 2:] - 2.0 * attn[..., 1:-1] + attn[..., :-2]  # [B,H,Q,N-2]
    if kv_mask is not None:
        m = kv_mask[:, None, None, 2:] * kv_mask[:, None, None, 1:-1] * kv_mask[:, None, None, :-2]
        d2 = d2 * m
        denom = m.sum()
    else:
        denom = d2.size
    return jnp.sum(d2 ** 2) / (denom + 1e-9)

# -------------------------
# Encoders
# -------------------------
class TransformerBlock(nnx.Module):
    def __init__(self, emb_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, rngs=None):
        super().__init__()
        self.ln1 = nnx.LayerNorm(emb_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=emb_dim, qkv_features=emb_dim, out_features=emb_dim,
            dropout_rate=dropout, decode=False, rngs=rngs)
        self.ln2 = nnx.LayerNorm(emb_dim, rngs=rngs)
        hidden = int(emb_dim * mlp_ratio)
        self.ffn1 = nnx.Linear(emb_dim, hidden, rngs=rngs)
        self.ffn2 = nnx.Linear(hidden, emb_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, attn_mask=None, deterministic=True):
        h = self.ln1(x)
        h = self.attn(h, mask=attn_mask, deterministic=deterministic)
        x = x + self.dropout(h, deterministic=deterministic)
        h = self.ln2(x)
        h = jax.nn.gelu(self.ffn1(h))
        h = self.ffn2(h)
        x = x + self.dropout(h, deterministic=deterministic)
        return x

class StateEncoder(nnx.Module):
    def __init__(self, in_dim: int, emb_dim: int, *, num_heads: int, n_layers: int = 3,
                 dropout_rate: float = 0.0, use_rope: bool = False, rngs=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nnx.Linear(in_dim, emb_dim, rngs=rngs)
        # self.blocks = [TransformerBlock(emb_dim, num_heads, dropout=dropout_rate, rngs=rngs) for _ in range(n_layers)]
        self.blocks = nnx.Dict(**{f"block_{i}": TransformerBlock(emb_dim, num_heads, dropout=dropout_rate, rngs=rngs)
                          for i in range(n_layers)})
        self.use_rope = use_rope  # 目前 RoPE 主要接在 CrossEncoder 中
        self.ln_out = nnx.LayerNorm(emb_dim, rngs=rngs)

    def __call__(self, states: jnp.ndarray, states_mask: jnp.ndarray, pos_start: int,
                 deterministic: bool = True):
        x = self.proj(states)  # [B,T,E]
        B, T, E = x.shape
        pos = jnp.arange(T, dtype=jnp.float32)[None, :] + pos_start
        pos = posemb_sincos_any(pos, E)
        x = x + pos

        m = states_mask.astype(bool)
        attn_mask = nnx.make_attention_mask(m, m)

        for blk in self.blocks.values():
            x = blk(x, attn_mask, deterministic)
        return self.ln_out(x)  # [B,T,E]

class CrossEncoder(nnx.Module):
    def __init__(self, emd_dim: int, *, num_heads: int, dropout_rate: float = 0.0,
                 use_rope: bool = False, rope_base: float = 10_000.0, rngs=None):
        super().__init__()
        assert num_heads is not None and emd_dim % num_heads == 0, "emd_dim 必须是 num_heads 的整数倍"
        self.emd_dim = emd_dim
        self.num_heads = num_heads
        self.head_dim = emd_dim // num_heads
        self.use_rope = use_rope
        self.rope_base = rope_base

        self.q_proj = nnx.Linear(emd_dim, emd_dim, rngs=rngs)
        self.k_proj = nnx.Linear(emd_dim, emd_dim, rngs=rngs)
        self.v_proj = nnx.Linear(emd_dim, emd_dim, rngs=rngs)
        self.out_proj = nnx.Linear(emd_dim, emd_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self,
                 q: jnp.ndarray,                # [B, Q_T, E]
                 kv: jnp.ndarray,               # [B, N,   E]
                 q_mask: jnp.ndarray | None = None,   # [B, Q_T]  可选
                 kv_mask: jnp.ndarray | None = None,   # [B, N]    可选
                 *,
                 q_pos_start: int | jnp.ndarray = 0,   # 标量或 [B]
                 kv_pos_start: int | jnp.ndarray = 0,  # 标量或 [B]
                 q_pos_idx: jnp.ndarray | None = None, # [B, Q_T] 可选
                 kv_pos_idx: jnp.ndarray | None = None,# [B, N]   可选
                 deterministic: bool = True
                 ) -> tuple[jnp.ndarray, jnp.ndarray]:
        B, Q_T, E = q.shape
        N = kv.shape[1]
        H, Dh = self.num_heads, self.head_dim
        scale = 1.0 / jnp.sqrt(jnp.array(Dh, dtype=q.dtype))

        # 线性投影并分头
        qh = self.q_proj(q).reshape(B, Q_T, H, Dh).transpose(0, 2, 1, 3)  # [B,H,Q_T,Dh]
        kh = self.k_proj(kv).reshape(B, N,   H, Dh).transpose(0, 2, 1, 3) # [B,H,N,Dh]
        vh = self.v_proj(kv).reshape(B, N,   H, Dh).transpose(0, 2, 1, 3) # [B,H,N,Dh]

        # ----- RoPE（按样本位置）：对 Q/K 施加旋转 -----
        if self.use_rope:
            assert Dh % 2 == 0, "RoPE 需要 head_dim 为偶数"

            # 计算每个样本、每个时间步的绝对位置索引
            if q_pos_idx is None:
                if jnp.ndim(q_pos_start) == 0:
                    q_pos_idx = jnp.arange(Q_T, dtype=jnp.int32)[None, :] + int(q_pos_start)           # [1,Q_T]
                    q_pos_idx = jnp.broadcast_to(q_pos_idx, (B, Q_T))
                else:
                    q_pos_idx = q_pos_start[:, None] + jnp.arange(Q_T, dtype=jnp.int32)[None, :]       # [B,Q_T]
            if kv_pos_idx is None:
                if jnp.ndim(kv_pos_start) == 0:
                    kv_pos_idx = jnp.arange(N, dtype=jnp.int32)[None, :] + int(kv_pos_start)           # [1,N]
                    kv_pos_idx = jnp.broadcast_to(kv_pos_idx, (B, N))
                else:
                    kv_pos_idx = kv_pos_start[:, None] + jnp.arange(N, dtype=jnp.int32)[None, :]       # [B,N]

            # 生成 cos/sin，并广播到 (B,H,T,Dh)
            cos_q, sin_q = rope_cos_sin_from_pos(q_pos_idx, Dh, base=self.rope_base, dtype=qh.dtype)   # [B,Q_T,Dh]
            cos_k, sin_k = rope_cos_sin_from_pos(kv_pos_idx, Dh, base=self.rope_base, dtype=kh.dtype)  # [B,N,  Dh]
            cos_q = cos_q[:, None, :, :]  # [B,1,Q_T,Dh] -> 广播到 H
            sin_q = sin_q[:, None, :, :]
            cos_k = cos_k[:, None, :, :]  # [B,1,N,Dh]
            sin_k = sin_k[:, None, :, :]

            qh = rope_apply(qh, cos_q, sin_q)  # [B,H,Q_T,Dh]
            kh = rope_apply(kh, cos_k, sin_k)  # [B,H,N,Dh]

        # 注意力
        scores = jnp.matmul(qh, jnp.swapaxes(kh, -1, -2)) * scale        # [B,H,Q_T,N]

        # KV mask：屏蔽不可见的 key/value
        if kv_mask is not None:
            kvm = kv_mask[:, None, None, :].astype(bool)                 # [B,1,1,N]
            scores = jnp.where(kvm, scores, jnp.full_like(scores, -1e9))

        attn = jax.nn.softmax(scores, axis=-1)                           # [B,H,Q_T,N]
        attn = self.dropout(attn, deterministic=deterministic)
        if kv_mask is not None:
            attn = attn * kvm
            denom = attn.sum(-1, keepdims=True) + 1e-9
            attn = attn / denom

        ctx_h = jnp.matmul(attn, vh)                                     # [B,H,Q_T,Dh]
        ctx = ctx_h.transpose(0, 2, 1, 3).reshape(B, Q_T, E)             # [B,Q_T,E]
        ctx = self.out_proj(ctx)                                         # [B,Q_T,E]

        # Q mask：把 padding 的 query 输出置零
        if q_mask is not None:
            ctx = ctx * q_mask[:, :, None]

        return ctx, attn

def safe_logvar(raw, logvar_min=-6.0):        # -6 比 -10 更常见，方差≈0.0025
    return jnp.log(jax.nn.softplus(raw)) + logvar_min

# -------------------------
# 潜变量头 & 解码器
# -------------------------
class GaussianHead(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int]=None, rngs=None):
        super().__init__()
        h = hidden or max(in_dim, out_dim)
        self.fc1 = nnx.Linear(in_dim, h, rngs=rngs)
        self.fc2 = nnx.Linear(h, h, rngs=rngs)
        self.mu_head = nnx.Linear(h, out_dim, rngs=rngs)
        self.lv_head = nnx.Linear(h, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        h = jax.nn.gelu(self.fc1(x))
        h = jax.nn.gelu(self.fc2(h))
        mu = self.mu_head(h)
        # logvar = jnp.clip(self.lv_head(h), -10.0, 5.0)  # 数值稳定
        logvar = safe_logvar(self.lv_head(h))
        return mu, logvar

class GaussianDecoderMLP(nnx.Module):
    def __init__(self, ctx_dim: int, z_dim: int, out_dim: int, hidden: Optional[int]=None, rngs=None):
        super().__init__()
        in_dim = ctx_dim + z_dim
        h = hidden or max(in_dim, 2*out_dim)
        self.fc1 = nnx.Linear(in_dim, h, rngs=rngs)
        self.fc2 = nnx.Linear(h, h, rngs=rngs)
        self.mu_head = nnx.Linear(h, out_dim, rngs=rngs)
        self.lv_head = nnx.Linear(h, out_dim, rngs=rngs)

    def __call__(self, ctx: jnp.ndarray, z: jnp.ndarray):
        x = jnp.concatenate([ctx, z], axis=-1)         # [B,T,E+Z]
        h = jax.nn.gelu(self.fc1(x))
        h = jax.nn.gelu(self.fc2(h))
        mu = self.mu_head(h)
        # logvar = jnp.clip(self.lv_head(h), -10.0, 5.0)
        logvar = safe_logvar(self.lv_head(h))
        return mu, logvar

class GaussianDecoderResidual(nnx.Module):
    def __init__(self, ctx_dim, z_dim, out_dim, hidden=None, rngs=None):
        super().__init__()
        h = hidden or max(ctx_dim + z_dim, 2*out_dim)
        # ctx 基线
        self.ctx_fc1 = nnx.Linear(ctx_dim, h, rngs=rngs)
        self.ctx_fc2 = nnx.Linear(h, h, rngs=rngs)
        self.mu_ctx  = nnx.Linear(h, out_dim, rngs=rngs)
        self.lv_ctx  = nnx.Linear(h, out_dim, rngs=rngs)
        # z 残差
        self.z_fc1   = nnx.Linear(z_dim, h, rngs=rngs)
        self.z_fc2   = nnx.Linear(h, h, rngs=rngs)
        self.mu_res  = nnx.Linear(h, out_dim, rngs=rngs)
        self.lv_res  = nnx.Linear(h, out_dim, rngs=rngs)
        self.film = nnx.Linear(z_dim, 2*h, rngs=rngs)

    def __call__(self, ctx, z):
        # ctx path
        hc = jax.nn.gelu(self.ctx_fc1(ctx))
        hc = jax.nn.gelu(self.ctx_fc2(hc))
        # z path
        hz = jax.nn.gelu(self.z_fc1(z))
        hz = jax.nn.gelu(self.z_fc2(hz))
        scale, shift = jnp.split(self.film(z), 2, axis=-1)
        hc_mod = hc * (1 + scale) + shift

        mu_base     = self.mu_ctx(hc_mod)
        logvar_base = self.lv_ctx(hc_mod)
        mu          = mu_base + self.mu_res(hz)
        logvar      = logvar_base + self.lv_res(hz)
        # logvar = jnp.clip(logvar, -10.0, 5.0)  # 或 softplus 参数化
        logvar = safe_logvar(logvar)
        return mu, logvar

class CondEmbed(nnx.Module):
    def __init__(self, *, latent_dim, action_dim,
                 emb_dim, num_cond_tokens=2, drop_p=0.2, rngs: nnx.Rngs):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_cond_tokens = num_cond_tokens
        self.drop_p = drop_p
        self.in_dim = latent_dim + 2*action_dim

        self.ln   = nnx.LayerNorm(self.in_dim, rngs=rngs)
        self.fc1  = nnx.Linear(self.in_dim, emb_dim, rngs=rngs)
        self.fc2  = nnx.Linear(emb_dim, num_cond_tokens * emb_dim, rngs=rngs)
        self.drop = nnx.Dropout(rate=drop_p)

    def __call__(self, z, mu_noise, logvar_noise, *, rngs: nnx.Rngs, train: bool):
        """
        z, mu_noise, logvar_noise: [B, ah, *]
        t: [B] 或 [B,1,1]
        return:
          cond_tokens: [B, K, emb_dim]
          cond_mask:   [B, K] (bool)
        """
        B, ah = z.shape[:2]

        feat = jnp.concatenate([z, mu_noise, logvar_noise], axis=-1)  # [B, ah, F]
        feat = self.ln(feat)                                               # [B, ah, F]

        # 池化成全局条件向量（也可替换为 attention pooling）
        cond_vec = jnp.mean(feat, axis=1)                                  # [B, F]

        # 小 MLP -> K*E
        h = jax.nn.relu(self.fc1(cond_vec))                                # [B, 2E]
        cond_flat = self.fc2(h)                                            # [B, K*E]
        cond_tokens = cond_flat.reshape(B, self.num_cond_tokens, self.emb_dim)  # [B, K, E]

        # 条件 dropout（整段 token 的按样本屏蔽）
        if train:
            # 这里用 Dropout 的“按元素”mask；想“整段屏蔽”可改为样本级 Bernoulli 再广播
            cond_tokens = self.drop(cond_tokens, rngs=nnx.Rngs(dropout=rngs), deterministic=not train)

        cond_mask = jnp.ones((B, self.num_cond_tokens), dtype=bool)
        return cond_tokens, cond_mask

class CondEmbedRNN(nnx.Module):
    def __init__(self, *, latent_dim, action_dim, emd_dim, num_cond_tokens=2, rnn_hidden=512, bidir=True, drop_p=0.1, rngs=None):
        super().__init__()
        self.bidir = bidir
        self.num_cond_tokens = num_cond_tokens
        self.in_dim = latent_dim + 2*action_dim
        self.emb_dim = emd_dim
        self.rnn_hidden = rnn_hidden
        self.proj_in = nnx.Linear(self.in_dim, rnn_hidden, rngs=rngs)
        self.gru_f = nnx.GRUCell(rnn_hidden, rnn_hidden, rngs=rngs)
        self.gru_b = nnx.GRUCell(rnn_hidden, rnn_hidden, rngs=rngs) if bidir else None
        self.ln = nnx.LayerNorm(rnn_hidden * (2 if bidir else 1), rngs=rngs)
        self.fc1 = nnx.Linear(rnn_hidden * (2 if bidir else 1), emd_dim, rngs=rngs)
        self.fc2 = nnx.Linear(emd_dim, num_cond_tokens * emd_dim, rngs=rngs)
        self.drop = nnx.Dropout(rate=drop_p, rngs=rngs)

    def __call__(self, z, mu_noise, logvar_noise, *, rngs: nnx.Rngs, train: bool):
        B, T, _ = z.shape
        h_f = jnp.zeros((B, self.rnn_hidden))
        h_b = jnp.zeros((B, self.rnn_hidden)) if self.bidir else None

        feat = jnp.concatenate([z, mu_noise, logvar_noise], axis=-1)  # [B, ah, F]
        x = self.proj_in(feat)  # [B, T, H]

        for t in range(T):
            h_f, _ = self.gru_f(x[:, t, :], h_f)
        if self.bidir:
            for t in reversed(range(T)):
                h_b, _ = self.gru_b(x[:, t, :], h_b)
        h = jnp.concatenate([h_f, h_b], axis=-1) if self.bidir else h_f
        h = self.ln(h)
        h = jax.nn.relu(self.fc1(h))                                # [B, E]

        if train:
            h = self.drop(h, rngs=nnx.Rngs(dropout=rngs), deterministic=not train)
        cond = self.fc2(h).reshape(B, self.num_cond_tokens, self.emb_dim)      # [B,K,E]
        cond_mask = jnp.ones((B, self.num_cond_tokens), dtype=bool)
        return cond, cond_mask

from typing import Literal, Tuple, Optional

def masked_mean(x: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    """x:[B,T,E], m:[B,T] -> [B,E]"""
    w = m.astype(x.dtype)[..., None]
    return (x * w).sum(axis=1) / (w.sum(axis=1) + 1e-8)

class CrossGate(nnx.Module):
    """
    对 cross-attn 输出做门控: ctx = base + g * (cross - base)

    mode:
      - "global": 每个样本一个 gate 标量 g ∈ [0,1], 形状 [B,1,1]（稳，推荐）
      - "per_step": 每个时间步一个 gate g_t ∈ [0,1], 形状 [B,T,1]（表达力强）

    超参:
      - hidden_mult: gate MLP 隐层宽度倍率（相对 E）
      - drop_rate:   样本级 gate-drop 概率（训练时随机将 g->0）
      - temp:        gate 温度（越小越接近硬）
      - sparsity:    稀疏正则系数，对 E[g] 做 L1（让不必要时 g 更小）

    输入:
      - his_e:[B,T,E], human_e:[B,Th,E], his_mask:[B,T], human_mask:[B,Th]
      - base_ctx:[B,T,E]   （一般是 his_e）
      - cross_ctx:[B,T,E]  （cross-enc 的输出）

    返回:
      - ctx_gated:[B,T,E]
      - g: [B,1,1] (global) 或 [B,T,1] (per_step)
      - reg: 标量正则项（稀疏 + gate-drop 的 KL 等，可自行扩展，这里仅 L1）
    """
    mode: Literal["global", "per_step"] = "global"
    hidden_mult: float = 0.5
    drop_rate: float = 0.0
    temp: float = 1.0
    sparsity: float = 0.0

    def __init__(
        self,
        *,
        embed_dim: int,          # E
        mode: Literal["global","per_step"]="global",
        hidden_mult: float = 0.5,
        drop_rate: float = 0.0,
        temp: float = 1.0,
        sparsity: float = 0.0,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_mult = hidden_mult
        self.drop_rate = drop_rate
        self.temp = temp
        self.sparsity = sparsity
        H = max(1, int(embed_dim * hidden_mult))

        if mode == "global":
            # 输入特征 3E → H
            self.fc1 = nnx.Linear(in_features=3*embed_dim, out_features=H, rngs=rngs)
            # H → 1
            self.fc2 = nnx.Linear(in_features=H, out_features=1, rngs=rngs)
        else:
            # per_step 一样是 3E → H → 1
            self.fc1 = nnx.Linear(in_features=3*embed_dim, out_features=H, rngs=rngs)
            self.fc2 = nnx.Linear(in_features=H, out_features=1, rngs=rngs)

        self.drop = nnx.Dropout(rate=self.drop_rate)

    def __call__(
        self,
        his_e: jnp.ndarray,            # [B,T,E]
        human_e: jnp.ndarray,          # [B,Th,E]
        his_mask: jnp.ndarray,         # [B,T]
        human_mask: jnp.ndarray,       # [B,Th]
        base_ctx: jnp.ndarray,         # [B,T,E]
        cross_ctx: jnp.ndarray,        # [B,T,E]
        *,
        rngs: nnx.Rngs,
        train: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        B, T, E = his_e.shape

        # 1) 构造 gate 的输入特征
        hq = masked_mean(his_e,   his_mask)          # [B,E]
        hk = masked_mean(human_e, human_mask)        # [B,E]
        if self.mode == "global":
            feat = jnp.concatenate([hq, hk, jnp.abs(hq - hk)], axis=-1)  # [B,3E]
            h = jax.nn.relu(self.fc1(feat))                              # [B,H]
            logit = self.fc2(h)                                          # [B,1]
            g = jax.nn.sigmoid(logit / self.temp)                        # [B,1]
            g = g[..., None]                                             # [B,1,1]
        else:
            # per_step：广播 human 池化向量到 T，再与 his_e 逐步拼接
            hk_T = jnp.broadcast_to(hk[:, None, :], (B, T, E))           # [B,T,E]
            feat = jnp.concatenate([his_e, hk_T, jnp.abs(his_e - hk_T)], axis=-1)  # [B,T,3E]
            h = jax.nn.relu(self.fc1(feat))                              # [B,T,H]
            logit = self.fc2(h)                                          # [B,T,1]
            g = jax.nn.sigmoid(logit / self.temp)                        # [B,T,1]

        # 2) gate-drop（样本级；训练期生效）
        if train and self.drop_rate > 0.0:
            key = rngs.dropout()
            if self.mode == "global":
                keep = (jax.random.uniform(key, (B, 1, 1)) > self.drop_rate).astype(g.dtype)
            else:
                # 逐步门控也用样本级随机关断，更稳；如需逐步关断可改成 (B,T,1)
                keep = (jax.random.uniform(key, (B, 1, 1)) > self.drop_rate).astype(g.dtype)
            g = g * keep

        # 3) 应用门控
        ctx = base_ctx + g * (cross_ctx - base_ctx)   # [B,T,E]

        # 4) 稀疏正则（可选）：L1 惩罚 E[g]
        reg = jnp.array(0.0, dtype=ctx.dtype)
        if self.sparsity > 0.0:
            if self.mode == "global":
                reg = self.sparsity * jnp.mean(jnp.abs(g))               # 标量
            else:
                # mask-aware：只在有效步上计算平均 gate
                w = his_mask[..., None].astype(g.dtype)                  # [B,T,1]
                reg = self.sparsity * jnp.sum(jnp.abs(g) * w) / (jnp.sum(w) + 1e-8)

        return ctx, g, reg
