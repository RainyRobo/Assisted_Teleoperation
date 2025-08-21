import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.attention import (
    make_attention_mask, make_causal_mask, combine_masks, dot_product_attention
)


# =============== 实用函数：滑窗 + 掩码 =================
def sliding_chunks_with_mask(
    x: jnp.ndarray,
    window: int,
    stride: int = 1,
    pad: bool = False,
    merge_batch: bool = False,
):
    """
    参数:
      x: [T, D] 或 [B, T, D]
      window: 窗口长度
      stride: 步长
      pad: 是否在末尾补零以对齐
      merge_batch: 是否把批次维和块数维合并 (返回 [B*N, window, D] / [B*N, window])

    返回:
      若输入是 [B, T, D]:
        chunks: [B, N, window, D] (或 merge 后 [B*N, window, D])
        mask:   [B, N, window]    (或 merge 后 [B*N, window])
      若输入是 [T, D]，对应去掉最前面的 B 维。
    """
    if x.ndim == 2:
        x = x[None, ...]  # 添加批次维
        squeeze_out = True
    elif x.ndim == 3:
        squeeze_out = False
    else:
        raise AssertionError(f"x should be [T, D] or [B, T, D], got {x.shape}")

    B, T0, D = x.shape
    T = T0
    pad_len = 0

    if pad:
        if T < window:
            pad_len = window - T
        else:
            remainder = (T - window) % stride
            pad_len = 0 if remainder == 0 else (stride - remainder)
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))
            T = x.shape[1]

    # 没法取窗口时，返回空
    if T < window:
        empty_chunks = jnp.empty((B, 0, window, D), dtype=x.dtype)
        empty_mask = jnp.empty((B, 0, window), dtype=jnp.bool_)
        if squeeze_out:
            return empty_chunks[0], empty_mask[0]
        return empty_chunks, empty_mask

    N = 1 + (T - window) // stride
    starts = jnp.arange(N) * stride  # [N]

    # 单个样本上抽取所有 chunk: [T, D] -> [N, window, D]
    def chunks_for_one_batch(x1):
        def take(start):
            return lax.dynamic_slice_in_dim(x1, start, window, axis=0)  # [window, D]
        return jax.vmap(take)(starts)

    # 对批次做 vmap: [B, T, D] -> [B, N, window, D]
    chunks = jax.vmap(chunks_for_one_batch)(x)

    # 生成有效位掩码（对所有 batch 相同；如果每个样本 T 不同，需要额外传入长度向量来做逐样本 mask）
    valid_vec = (
        jnp.concatenate(
            [jnp.ones((T0,), dtype=jnp.bool_), jnp.zeros((pad_len,), dtype=jnp.bool_)],
            axis=0,
        )
        if pad else jnp.ones((T,), dtype=jnp.bool_)
    )

    def take_mask(start):
        return lax.dynamic_slice_in_dim(valid_vec, start, window, axis=0)  # [window]

    base_mask = jax.vmap(take_mask)(starts)             # [N, window]
    mask = jnp.broadcast_to(base_mask, (B, N, window))  # [B, N, window]

    # 是否展平批次和块数
    if merge_batch:
        chunks = chunks.reshape(B * N, window, D)
        mask = mask.reshape(B * N, window)

    # 如果输入是 [T, D]，去掉批次维
    if squeeze_out and not merge_batch:
        return chunks[0], mask[0]
    return chunks, mask


# =============== 自注意力（按 chunk） ====================
class ChunkSelfAttention(nn.Module):
    """对每个 chunk 独立做 Self-Attention。若需要改变输出维度，设置 out_features。"""
    num_heads: int
    dropout_rate: float = 0.0
    causal: bool = False
    out_features: int | None = None  # 兼容性更好：不用在 SelfAttention 里传 out_features

    @nn.compact
    def __call__(self, x, mask=None, deterministic: bool = True):
        """
        支持两种输入：
          x:    [B, T, D] 或 [B, N, T, D]  (N=chunk_num)
          mask: [B, T]   或 [B, N, T]     (bool) True=有效，False=padding
        返回形状与 x 一致（最后一维可能因 out_features 改变）
        """
        if x.ndim == 3:
            B, T, D = x.shape
            BN = B
            x_eff = x                       # [B, T, D]
            mask_eff = mask                 # [B, T] 或 None
        elif x.ndim == 4:
            B, N, T, D = x.shape
            BN = B * N
            x_eff = x.reshape(BN, T, D)     # [B*N, T, D]
            if mask is not None:
                assert mask.shape[:3] == (B, N, T), f"mask应为[B,N,T]，收到{mask.shape}"
                mask_eff = mask.reshape(BN, T).astype(jnp.bool_)  # [B*N, T]
            else:
                mask_eff = None
        else:
            raise AssertionError(f"x must be [B,T,D] or [B,N,T,D], got {x.shape}")

        assert D % self.num_heads == 0, "D 必须能被 num_heads 整除"

        # 1) 组装注意力掩码（padding + 可选 causal）
        attn_mask = None
        if mask_eff is not None:
            padding_mask = make_attention_mask(mask_eff, mask_eff, dtype=jnp.bool_)  # [BN, 1, T, T]
            if self.causal:
                causal_mask = make_causal_mask(jnp.ones((BN, T), dtype=jnp.bool_))   # [BN, 1, T, T]
                attn_mask = combine_masks(padding_mask, causal_mask, dtype=jnp.bool_)
            else:
                attn_mask = padding_mask
        elif self.causal:
            attn_mask = make_causal_mask(jnp.ones((BN, T), dtype=jnp.bool_))         # [BN, 1, T, T]

        # 2) Self-Attention（注意：必须以关键字参数传入 mask）
        y_eff = nn.SelfAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x_eff, mask=attn_mask)  # [BN, T, D]

        # 3) 可选的输出映射到指定维度
        F = self.out_features if (self.out_features is not None and self.out_features != D) else D
        if F != D:
            y_eff = nn.Dense(F)(y_eff)      # [BN, T, F]

        # 4) 还原形状
        if x.ndim == 4:
            y = y_eff.reshape(B, N, T, F)   # [B, N, T, F]
        else:
            y = y_eff                       # [B, T, F]
        return y


# =============== 跨注意力 + VAE 头（返回 z, mu, logvar, kl） ===============
class CrossAttentionOverChunksVAE(nn.Module):
    """
    与 CrossAttentionOverChunks 相同的跨注意力，但输出 (z, mu, logvar, kl_loss)。
    - 先对每个 chunk 做一次 cross-attn，再在 N 维聚合（sum/mean）
    - 合并 heads 后，用两条线性头得到 mu/logvar
    - 训练时重参数化采样，评估时直接用 mu
    - KL 对有效 token 做掩码并归一到“每个有效 token 的平均值”
    """
    num_heads: int
    latent_dim: int                 # VAE 的潜变量维度（每个 query 位置的 z 维度）
    dropout_rate: float = 0.0
    aggregate: str = "mean"         # "sum" 或 "mean"
    clip_logvar: tuple[float, float] = (-10.0, 10.0)
    beta_kl: float = 1.0            # β-VAE 系数

    @nn.compact
    def __call__(self, q, kv, q_mask=None, kv_mask=None, deterministic: bool = True):
        """
        q:       [B, Tq, D]
        kv:      [B, N, Tk, D]
        q_mask:  [B, Tq] (bool)
        kv_mask: [B, N, Tk] (bool)
        return:  (z, mu, logvar, kl_loss)
                 其中 z/mu/logvar 形状都是 [B, Tq, latent_dim]，kl_loss 是标量
        """
        assert q.ndim == 3 and kv.ndim == 4, f"q {q.shape} 应为 [B,Tq,D]，kv {kv.shape} 应为 [B,N,Tk,D]"
        Bq, Tq, Dq = q.shape
        Bk, N, Tk, Dk = kv.shape
        assert Bq == Bk and Dq == Dk, "batch 或通道维不匹配"
        H = self.num_heads
        assert Dq % H == 0, "D 必须能被 num_heads 整除"
        Dh = Dq // H

        if q_mask is None:
            q_mask = jnp.ones((Bq, Tq), dtype=jnp.bool_)
        if kv_mask is None:
            kv_mask = jnp.ones((Bk, N, Tk), dtype=jnp.bool_)

        # 1) Q/K/V 投影（分头）
        q_proj = nn.DenseGeneral((H, Dh), name="q_proj")(q)      # [B, Tq, H, Dh]
        k_proj = nn.DenseGeneral((H, Dh), name="k_proj")(kv)     # [B, N, Tk, H, Dh]
        v_proj = nn.DenseGeneral((H, Dh), name="v_proj")(kv)     # [B, N, Tk, H, Dh]

        def attn_one(k_one, v_one, mask_one):
            # k_one/v_one: [B, Tk, H, Dh], mask_one: [B, Tk] (bool)
            attn_mask = make_attention_mask(q_mask, mask_one, dtype=jnp.bool_)     # [B,1,Tq,Tk]
            ctx = dot_product_attention(
                query=q_proj, key=k_one, value=v_one,
                mask=attn_mask,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                dtype=q.dtype,
                precision=None,
            )  # [B, Tq, H, Dh]
            return ctx

        # 2) 在 N 维上做 vmap（逐块 attention），再聚合
        ctx_all = jax.vmap(attn_one, in_axes=(1, 1, 1), out_axes=1)(k_proj, v_proj, kv_mask)  # [B, N, Tq, H, Dh]
        if self.aggregate == "mean":
            ctx = ctx_all.mean(axis=1)  # [B, Tq, H, Dh]
        else:
            ctx = ctx_all.sum(axis=1)   # [B, Tq, H, Dh]
            
            
            
            

        # # 3) 合并 heads，并产生 mu / logvar
        # token_feat = nn.DenseGeneral(Dq, axis=(-2, -1), name="out_proj")(ctx)  # [B, Tq, D]
        # mu      = nn.Dense(self.latent_dim, name="mu_head")(token_feat)        # [B, Tq, Z]
        # logvar  = nn.Dense(self.latent_dim, name="logvar_head")(token_feat)    # [B, Tq, Z]
        # logvar  = jnp.clip(logvar, *self.clip_logvar)

        # # 4) 重参数化（训练采样，评估用均值）
        # if deterministic:
        #     z = mu
        # else:
        #     eps = jax.random.normal(self.make_rng("reparam"), mu.shape, dtype=mu.dtype)
        #     std = jnp.exp(0.5 * logvar)
        #     z = mu + std * eps

        # # 5) 计算 KL(q(z|x)||N(0,1))，对有效 token 做掩码，并做归一化（平均到有效 token）
        # kl = 0.5 * (jnp.exp(logvar) + mu**2 - 1.0 - logvar)       # [B, Tq, Z]
        # m = q_mask[..., None].astype(mu.dtype)                     # [B, Tq, 1]
        # kl_masked = (kl * m).sum()                                 # 标量（批次 + 时序 + 维度求和）
        # denom = jnp.maximum(m.sum() * self.latent_dim, 1.0)        # 有效元素个数，避免除 0
        # kl_loss = self.beta_kl * (kl_masked / denom)

        return z, mu, logvar, kl_loss


# ===================== 演示流程 ===========================
if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)

    # 1) 生成 human 动作并切成 chunk + mask
    batch_size = 28
    Human_Action_num = 150
    action_dim = 32
    Human_Action = jax.random.normal(rng, (batch_size, Human_Action_num, action_dim))
    print("\n human action:", Human_Action.shape)  # [B, T, D]
    human_chunks, human_mask = sliding_chunks_with_mask(
        Human_Action, window=80, stride=10, pad=True
    )
    print("human_chunks:", human_chunks.shape)  # [B, N, 80, 32]
    print("human_mask:", human_mask.shape)      # [B, N, 80]
    
    #========================初始化模型========================
    num_heads = 4
    dropout_rate = 0.0
    out_features = 64
    
    # 同一个模块可复用结构，但通常论文里 human/state 会各自一套参数；这里为演示各自单独 init
    self_attn_h = ChunkSelfAttention(num_heads, dropout_rate, causal=False, out_features=out_features)
    self_attn_s = ChunkSelfAttention(num_heads, dropout_rate, causal=False, out_features=out_features)
    cross_vae   = CrossAttentionOverChunksVAE(num_heads, latent_dim=64, dropout_rate=dropout_rate, aggregate="mean")

    # ======================== 自注意力：human（逐 chunk） ========================
    rng_params_h = jax.random.PRNGKey(42)
    variables_sa_h = self_attn_h.init({'params': rng_params_h}, human_chunks, mask=human_mask, deterministic=True)
    human_emb = self_attn_h.apply(variables_sa_h, human_chunks, mask=human_mask, deterministic=True)  # [B, N, 80, 64]
    print("human_emb:", human_emb.shape)
    
    # 2) 生成 state 并做自注意力（全序列）
    state = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 50, action_dim))
    state_mask = jnp.ones((batch_size, 50), dtype=jnp.bool_)
    print("\n state:", state.shape)  # [B, T, D]
    
    variables_sa_s = self_attn_s.init({'params': jax.random.PRNGKey(43)}, state, mask=state_mask, deterministic=True)
    state_emb = self_attn_s.apply(variables_sa_s, state, mask=state_mask, deterministic=True)  # [B, 50, 64]
    print("state_emb:", state_emb.shape)

    # 3) 跨注意力：用 state_emb 作为 Q，对 human_emb 的每个 chunk 做查询，并在 N 上聚合
    vars_cross = cross_vae.init(
        {'params': jax.random.PRNGKey(44)},
        state_emb, human_emb, q_mask=state_mask, kv_mask=human_mask, deterministic=True
    )
    z, mu, logvar, kl_loss = cross_vae.apply(
        vars_cross,
        state_emb, human_emb, q_mask=state_mask, kv_mask=human_mask, deterministic=True
    )
    print("z:", z.shape)        # [B, 50, 64]
    print("mu:", mu.shape)      # [B, 50, 64]
    print("logvar:", logvar.shape)  # [B, 50, 64]
    print("kl_loss:", float(kl_loss))
    
    # 训练时示例（需要 reparam 随机数）：
    # z_train, _, _, kl = cross_vae.apply(
    #     vars_cross,
    #     state_emb, human_emb, q_mask=state_mask, kv_mask=human_mask,
    #     deterministic=False,
    #     rngs={'reparam': jax.random.PRNGKey(123)}
    # )
