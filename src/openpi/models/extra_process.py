import jax
import jax.numpy as jnp
import flax.nnx as nnx

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

def masked_mean(x: jnp.ndarray, mask: jnp.ndarray, axis: int) -> jnp.ndarray:
    m = mask.astype(x.dtype)[..., None]
    s = (x * m).sum(axis=axis)
    d = jnp.clip(m.sum(axis=axis), a_min=1e-6)
    return s / d

class HumanChunkEncoderBN(nnx.Module):
    """
    输入:
      human_chunks: [B, N, T, D]
      human_mask:   [B, N, T]  (bool) True=有效
      stride: int   生成全局绝对位置用: pos = chunk_id*stride + offset
    输出:
      pooled: [B, N, E]
      chunk_mask: [B, N]  (该 chunk 是否至少有一个有效 token)
    """
    def __init__(self,
                 in_dim: int,
                 emb_dim: int,
                 *,
                 num_heads: int | None = None,
                 dropout_rate: float = 0.0,
                 use_chunk_attn: bool = False,
                 reduce: str = "mean",   # 'mean' | 'last' | 'max' | 'attn' | 'cls'
                 rngs=None):
        self.proj = nnx.Linear(in_dim, emb_dim, rngs=rngs)
        self.use_chunk_attn = use_chunk_attn and (num_heads is not None)
        self.reduce = reduce
        if self.use_chunk_attn:
            self.attn = nnx.MultiHeadAttention(num_heads=num_heads, 
                                               in_features=emb_dim,
                                               qkv_features=emb_dim,
                                               out_features=emb_dim,
                                               dropout_rate=dropout_rate, 
                                               rngs=rngs,
                                               decode=False)
        else:
            self.attn = None

        # attention pooling（可学习权重），仅在 reduce='attn' 时用
        if reduce == "attn":
            self.pool_v = nnx.Linear(emb_dim, 1, rngs=rngs)  # 分数 s_t = v^T h_t
        else:
            self.pool_v = None

        # CLS token（只在 reduce='cls' 时用，需 self.attn）
        if reduce == "cls":
            if not self.use_chunk_attn:
                raise ValueError("reduce='cls' 需要 use_chunk_attn=True 以便在 chunk 内跑一次 self-attn")
            # 一个可学习的 CLS 向量
            self.cls = nnx.Param(jnp.zeros((1, 1, 1, emb_dim), dtype=jnp.float32))

    def __call__(self,
                 human_chunks: jnp.ndarray,
                 human_mask: jnp.ndarray,
                 *,
                 stride: int,
                 deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        B, N, T, D = human_chunks.shape
        E = self.proj.out_features

        # 1) 线性投影到 emb
        x = human_chunks.reshape(B * N, T, D)    # (B*N, T, D)
        x = self.proj(x)                         # (B*N, T, E)

        # 2) 绝对位置编码（全局 index = chunk_id*stride + offset）
        pos_idx = (jnp.arange(N)[None, :, None] * stride) + jnp.arange(T)[None, None, :]  # (1,N,T)
        pos_idx = jnp.broadcast_to(pos_idx, (B, N, T)).reshape(B * N, T)                  # (B*N, T)
        pos_emb = posemb_sincos_any(pos_idx, E)                                           # (B*N, T, E)
        x = x + pos_emb

        # 3) 可选：chunk 内 self-attention
        m_eff = human_mask.reshape(B * N, T)  # (B*N, T)
        if self.attn is not None:
            attn_mask = nnx.make_attention_mask(m_eff, m_eff)  # [B*N, 1, T, T]
            x = self.attn(x, mask=attn_mask, deterministic=deterministic, decode=False)  # (B*N, T, E)

        # 4) 在 T 维上汇聚 -> (B*N, E)
        if self.reduce == "mean":
            pooled = masked_mean(x, m_eff, axis=1)
        elif self.reduce == "max":
            neg_inf = jnp.finfo(x.dtype).min
            x_masked = jnp.where(m_eff[..., None], x, neg_inf)
            pooled = x_masked.max(axis=1)
        elif self.reduce == "last":
            # 取每个 chunk 的最后一个有效步（全 pad 则回退到 0 向量）
            valid_len = m_eff.sum(axis=1)                  # (B*N,)
            last_idx  = jnp.clip(valid_len - 1, 0)         # (B*N,)
            gather_idx = last_idx[:, None, None].repeat(E, axis=2)  # (B*N,1,E)
            last_tok = jnp.take_along_axis(x, gather_idx, axis=1).squeeze(1)  # (B*N,E)
            pooled = jnp.where((valid_len > 0)[:, None], last_tok, jnp.zeros_like(last_tok))
        elif self.reduce == "attn":
            # learnable attention pooling: softmax(v^T h_t)
            scores = self.pool_v(x).squeeze(-1)            # (B*N, T)
            scores = jnp.where(m_eff, scores, -1e30)       # 遮掉 padding
            alpha  = jax.nn.softmax(scores, axis=1)        # (B*N, T)
            pooled = (x * alpha[..., None]).sum(axis=1)    # (B*N, E)
        elif self.reduce == "cls":
            # prepend CLS，mask=True，跑一次 attn，取 CLS 输出
            cls_tok  = jnp.broadcast_to(self.cls.value, (B, N, 1, E)).reshape(B*N, 1, E)  # (B*N,1,E)
            x_cls    = jnp.concatenate([cls_tok, x], axis=1)                               # (B*N,1+T,E)
            m_cls    = jnp.concatenate([jnp.ones((B*N,1), dtype=jnp.bool_), m_eff], axis=1)
            attn_mask = nnx.make_attention_mask(m_cls, m_cls)
            x_cls    = self.attn(x_cls, mask=attn_mask, deterministic=deterministic, decode=False)           # (B*N,1+T,E)
            pooled   = x_cls[:, 0, :]                                                      # (B*N,E)
        else:
            raise ValueError(f"Unknown reduce='{self.reduce}'")

        # 5) 还原到 (B, N, E)
        pooled = pooled.reshape(B, N, E)
        # 该 chunk 是否至少有一个有效 token
        chunk_mask = human_mask.any(axis=-1)  # (B, N)
        return pooled, chunk_mask



class StateSeqEncoderToChunk(nnx.Module):
    """
    输入:
      state:      [B, T, D]
      state_mask: [B, T] (可选，None 则全 True)
    输出:
      pooled:     [B, 1, E]  # 每个样本一个 state-chunk 向量
      chunk_mask: [B, 1]     # 该 chunk 是否有效（是否存在 True）
    """
    def __init__(self,
                 in_dim: int,
                 emb_dim: int,
                 *,
                 num_heads: int | None = None,
                 dropout_rate: float = 0.0,
                 use_self_attn: bool = False,
                 reduce: str = "mean",  # 'mean' | 'last' | 'max' | 'attn' | 'cls'
                 rngs=None):
        self.proj = nnx.Linear(in_dim, emb_dim, rngs=rngs)
        self.use_self_attn = use_self_attn and (num_heads is not None)
        self.reduce = reduce
        if self.use_self_attn:
            self.attn = nnx.MultiHeadAttention(num_heads=num_heads,
                                          in_features=emb_dim,    
                                          qkv_features=emb_dim,
                                          out_features=emb_dim,
                                          dropout_rate=dropout_rate,
                                          rngs=rngs, decode=False)
        else:
            self.attn = None
        if reduce == "attn":
            self.pool_v = nnx.Linear(emb_dim, 1, rngs=rngs)  # s_t = v^T h_t
        else:
            self.pool_v = None
        if reduce == "cls":
            if not self.use_self_attn:
                raise ValueError("reduce='cls' 需要 use_self_attn=True")
            self.cls = nnx.Param(jnp.zeros((1, 1, emb_dim), dtype=jnp.float32))  # (1,1,E)

    def __call__(self,
                 state: jnp.ndarray,
                 state_mask: jnp.ndarray | None = None,
                 *,
                 deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        B, T, D = state.shape
        E = self.proj.out_features

        x = self.proj(state)  # (B,T,E)

        # 绝对位置（0..T-1）
        pos_idx = jnp.arange(T)[None, :]               # (1,T)
        pos_emb = posemb_sincos_any(pos_idx, E)        # (1,T,E) -> 广播到 (B,T,E)
        x = x + pos_emb

        if state_mask is None:
            state_mask = jnp.ones((B, T), dtype=jnp.bool_)

        # 可选：序列自注意力
        if self.attn is not None:
            attn_mask = nnx.make_attention_mask(state_mask, state_mask)
            x = self.attn(x, mask=attn_mask, deterministic=deterministic, decode=False)  # (B,T,E)

        # 汇聚到单向量 (B,E)
        if self.reduce == "mean":
            pooled = masked_mean(x, state_mask, axis=1)
        elif self.reduce == "max":
            neg_inf = jnp.finfo(x.dtype).min
            x_masked = jnp.where(state_mask[..., None], x, neg_inf)
            pooled = x_masked.max(axis=1)
        elif self.reduce == "last":
            valid_len = state_mask.sum(axis=1)              # (B,)
            last_idx  = jnp.clip(valid_len - 1, 0)          # (B,)
            gather_idx = last_idx[:, None, None].repeat(E, axis=2)  # (B,1,E)
            last_tok = jnp.take_along_axis(x, gather_idx, axis=1).squeeze(1)  # (B,E)
            pooled = jnp.where((valid_len > 0)[:, None], last_tok, jnp.zeros_like(last_tok))
        elif self.reduce == "attn":
            scores = self.pool_v(x).squeeze(-1)           # (B,T)
            scores = jnp.where(state_mask, scores, -1e30)
            alpha  = jax.nn.softmax(scores, axis=1)
            pooled = (x * alpha[..., None]).sum(axis=1)
        elif self.reduce == "cls":
            cls_tok = jnp.broadcast_to(self.cls.value, (B, 1, E))  # (B,1,E)
            x_cls   = jnp.concatenate([cls_tok, x], axis=1)        # (B,1+T,E)
            m_cls   = jnp.concatenate([jnp.ones((B,1), dtype=jnp.bool_), state_mask], axis=1)
            x_cls   = self.attn(x_cls, mask=m_cls, deterministic=deterministic, decode=False)  # (B,1+T,E)
            pooled  = x_cls[:, 0, :]
        else:
            raise ValueError(f"Unknown reduce='{self.reduce}'")

        # 变成 (B,1,E)，并返回 chunk 级 mask
        pooled = pooled[:, None, :]                # (B,1,E) ✅
        chunk_mask = state_mask.any(axis=1, keepdims=True)  # (B,1)
        return pooled, chunk_mask
      
      
def take_prev_window_pad0(x: jnp.ndarray, idx: jnp.ndarray, window: int = 50):
    """
    取每个样本在 idx 之前的 window 帧；不够则左侧补 0。
    x   : [B, T, D]  序列
    idx : [B]        每个样本的“当前位置”，取 [idx-window, idx) 区间
    返回:
      y     : [B, window, D]  窗口数据（已补 0）
      valid : [B, window]     True=来自原始数据，False=补 0
    """
    B, T, D = x.shape
    # 把 idx 限制在 [0, T]，避免越界
    idx = jnp.clip(idx, 0, T)

    # 在时间维左侧 pad window 个 0，然后从位置 idx 开始切 window 长度
    x_pad = jnp.pad(x, ((0, 0), (window, 0), (0, 0)))  # [B, T+window, D]

    def slice_one(xp, i):
        # 从 xp[i : i+window] 切（注意 xp 的时间维已左移 window）
        return jax.lax.dynamic_slice_in_dim(xp, i, window, axis=0)  # [window, D]

    y = jax.vmap(slice_one, in_axes=(0, 0))(x_pad, idx)  # [B, window, D]

    # 有效位：对应原始下标 s = (idx - window) + offset 是否落在 [0, T)
    start = idx - window                   # [B]
    off   = jnp.arange(window)[None, :]    # [1, window]
    src   = start[:, None] + off           # [B, window]
    valid = (src >= 0) & (src < T)         # [B, window]

    return y, valid


class CrossAttnPool(nnx.Module):
    def __init__(self, emb_dim: int, num_heads: int = 8, dropout_rate: float = 0.0, rngs=None):
        assert emb_dim % num_heads == 0, "emb_dim 必须能被 num_heads 整除"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        # 线性投影到 Q/K/V，多头后再一个 out projection
        self.q_proj  = nnx.Linear(emb_dim, emb_dim, rngs=rngs)
        self.k_proj  = nnx.Linear(emb_dim, emb_dim, rngs=rngs)
        self.v_proj  = nnx.Linear(emb_dim, emb_dim, rngs=rngs)
        self.o_proj  = nnx.Linear(emb_dim, emb_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate)

    def __call__(self, q: jnp.ndarray, kv: jnp.ndarray, kv_mask: jnp.ndarray | None = None,
                 *, deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        q:       [B, 1, E]
        kv:      [B, N, E]
        kv_mask: [B, N] (bool)  True=有效
        return: (ctx: [B, 1, E], attn: [B, num_heads, 1, N])
        """
        B, _, E = q.shape
        N = kv.shape[1]
        H = self.num_heads
        Dh = self.head_dim

        # 线性投影
        qh = self.q_proj(q).reshape(B, 1, H, Dh).transpose(0, 2, 1, 3)   # [B,H,1,Dh]
        kh = self.k_proj(kv).reshape(B, N, H, Dh).transpose(0, 2, 1, 3)  # [B,H,N,Dh]
        vh = self.v_proj(kv).reshape(B, N, H, Dh).transpose(0, 2, 1, 3)  # [B,H,N,Dh]

        # 注意力分数
        scores = jnp.matmul(qh, jnp.swapaxes(kh, -1, -2)) / jnp.sqrt(Dh)  # [B,H,1,N]

        if kv_mask is not None:
            mask = kv_mask[:, None, None, :].astype(bool)                 # [B,1,1,N] -> broadcast
            scores = jnp.where(mask, scores, jnp.array(-1e30, scores.dtype))

        attn = jax.nn.softmax(scores, axis=-1)                            # [B,H,1,N]
        attn = self.dropout(attn, deterministic=deterministic)

        # 上下文
        ctx_h = jnp.matmul(attn, vh)                                      # [B,H,1,Dh]
        ctx = ctx_h.transpose(0, 2, 1, 3).reshape(B, 1, E)                # [B,1,E]
        ctx = self.o_proj(ctx)                                            # [B,1,E]
        return ctx, attn


class GaussianHeads(nnx.Module):
    def __init__(self, emb_dim: int, out_dim: int, rngs=None):
        self.mu_head     = nnx.Linear(emb_dim, out_dim, rngs=rngs)
        self.logvar_head = nnx.Linear(emb_dim, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        x: [B, 1, E] or [B, E]
        returns:
          mu:     [B, D]
          logvar: [B, D]
        """
        if x.ndim == 3:  # [B,1,E] -> [B,E]
            x = x.squeeze(axis=1)
        mu     = self.mu_head(x)
        logvar = self.logvar_head(x)
        # 可选：稳定化（避免过大过小）
        logvar = jnp.clip(logvar, -10.0, 10.0)
        return mu, logvar
