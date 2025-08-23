import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.models.extra_process as _extra_process


logger = logging.getLogger("openpi")
def sliding_chunks_with_mask(
    x: jnp.ndarray,
    window: int,
    stride: int = 1,
    pad: bool = False,
    merge_batch: bool = False,
):
    if x.ndim == 2:
        x = x[None, ...] 
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
            return jax.lax.dynamic_slice_in_dim(x1, start, window, axis=0)  # [window, D]
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
        return jax.lax.dynamic_slice_in_dim(valid_vec, start, window, axis=0)  # [window]

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



def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    latent_dim: int = 8

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                    "low_0_rgb": image_spec  # 新增
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                    "low_0_rgb":image_mask_spec
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        
        keep_new_modules = nnx_utils.PathRegex(
            r".*(state_enc|cross_enc|post_head|prior_head|decoder|cond_emd|cross_gate).*"
        )
        filters.append(nnx.Not(keep_new_modules))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        
        self.vae_hidden_dim = 128
        
        # encoders
        self.state_enc = _extra_process.StateEncoder(config.action_dim, self.vae_hidden_dim,
                                      num_heads=4, n_layers=2, rngs=rngs)
        self.cross_enc = _extra_process.CrossEncoder(self.vae_hidden_dim,
                                      num_heads=4, use_rope=True, rngs=rngs)

        # VAE heads
        self.post_head  = _extra_process.GaussianHead(self.vae_hidden_dim, config.latent_dim, rngs=rngs)  # q(z|ctx)
        self.prior_head = _extra_process.GaussianHead(self.vae_hidden_dim, config.latent_dim, rngs=rngs)  # p(z|his_e)
        self.decoder = _extra_process.GaussianDecoderResidual(self.vae_hidden_dim, config.latent_dim, config.action_dim, rngs=rngs)
        self.drop_rate = 0.1
        
        # 
        self.cond_emd = _extra_process.CondEmbed(latent_dim=config.latent_dim, action_dim=config.action_dim, emb_dim=2 * action_expert_config.width, num_cond_tokens=1, drop_p=0.2, rngs=rngs)
        self.cross_gate = _extra_process.CrossGate(embed_dim=self.vae_hidden_dim, mode="global", hidden_mult=0.5, drop_rate=0.1, temp=1.0, sparsity=1e-3, rngs=rngs,)
        self.consistency_prob = 0.3
        self.consistency_lambda = 0.01
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # 1. embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # 2. embed language
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        # 
        input_mask = []
        ar_mask = []
        tokens = []
        # 1. add a single state token
        # state_token shape: (b:28, 1, emb:1024)
        state_token = self.state_proj(obs.state)[:, None, :] # [b,s] -> [b, 1, s:32] -> [b,1,e]
        tokens.append(state_token)
        
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]
        
        # 2. add action tokens
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        
        # 把噪声强度 t 和 加噪动作 x_t 混在一起编码，确保明白动作是多大噪声下的样子
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        # swish 激活函数
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
    
    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation, 
        actions: _model.Actions, 
        *, 
        train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        
        # noise (28, 50, 32)
        noise = jax.random.normal(noise_rng, actions.shape) 
        
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        
        # ground truth!
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        # prefix_tokens including images and language: jnp.concatenate(img.emb, lang.emb,axis=1)
        # prefix_tokens (28, 816, 2048)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        #suffix_tokens including state and action_time token
        # suffix_tokens (28, 51, 1024)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        
        # (b,images+language+state+action_time,emb)
        # input_mask (28, 867)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        
        # ar_mask (867,)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        # suffix_out (10, 51, 1024)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        
        # suffix_out (10, 51, 1024)-> v_t (10, 50, 32)
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)
    
    @override
    def compute_loss_extra(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,         # [B, ah, D]
        human_action: dict,
        his_state: dict,
        *,
        kl_weight: float,
        tcc_weight: float,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:

        observation = _model.preprocess_observation(None, observation, train=False)
        rng, eps_z_rng, eps_pad_rng, time_rng, eps_noise_rng, eps_noise_pad_rng, drop_rng, cond_rng = jax.random.split(rng, 8)

        drop_flag = (jax.random.uniform(drop_rng, ()) < self.drop_rate)

        # ---- 数据 & mask ----
        human_action_data = human_action["data"]                    # [B, Th, D]
        his_state_data    = his_state["data"]                       # [B, Ts, D]
        human_mask        = human_action.get("mask", jnp.ones(human_action_data.shape[:2], bool))
        his_mask          = his_state.get("mask",   jnp.ones(his_state_data.shape[:2], bool))
        
        curr_id = his_state["curr_id"]
        his_window_size = 50

        his_state_data, his_mask = _extra_process.get_history_window(his_state_data, his_mask, curr_id, his_window_size)

        human_mask = human_mask.astype(bool)
        his_mask   = his_mask.astype(bool)
        has_human_batch = jnp.any(human_mask)
        # 训练时有一定概率丢弃 human
        use_human = jnp.logical_and(has_human_batch, jnp.logical_not(drop_flag))

        # ---- 编码到 E ----
        human_state_emd = self.state_enc(human_action_data, human_mask, pos_start=0, deterministic=not train)  # [B, Th, E]
        his_state_emd   = self.state_enc(his_state_data,    his_mask,   pos_start=0, deterministic=not train)  # [B, Ts, E]


        q_pos_idx = curr_id - his_window_size + 1
        q_pos_start = jnp.clip(q_pos_idx, 0, 1000)
        # jax.debug.print("q_pos_start = {x}", x=q_pos_start.shape)
        # ---- Cross-Attention：Q=his_e, KV=human_e ----
        def build_ctx_with_human(_):
            ctx, attn = self.cross_enc(his_state_emd, human_state_emd, kv_mask=human_mask, q_mask=his_mask,
                                    q_pos_start=q_pos_start, kv_pos_start=0, deterministic=not train)                 # ctx:[B, Ts, E]
           
            ctx_gated, g, reg_gate = self.cross_gate(
                his_e=his_state_emd, human_e=human_state_emd,
                his_mask=his_mask, human_mask=human_mask,
                base_ctx=his_state_emd, cross_ctx=ctx,
                rngs=nnx.Rngs(dropout=drop_rng), train=train
            )
            return ctx_gated, attn, g, reg_gate
        
        def build_ctx_without_human(_):
            dummy_attn = jnp.zeros((his_state_emd.shape[0], 4, his_state_emd.shape[1], human_state_emd.shape[1]))
            g = jnp.zeros((his_state_emd.shape[0], 1, 1), dtype=his_state_emd.dtype)
            reg_gate = jnp.array(0.0, dtype=his_state_emd.dtype)  
            return his_state_emd, dummy_attn, g, reg_gate
        
        ctx, attn, gate_g, reg_gate = jax.lax.cond(use_human, build_ctx_with_human, build_ctx_without_human, operand=None)

        # 对齐 action_horizon
        ah = actions.shape[1]
        Ts = ctx.shape[1]
        assert Ts == ah, f"CrossEncoder 输出长度 Ts={Ts} 必须与 action_horizon ah={ah} 对齐。"

        # ---- VAE: q(z|ctx), p(z|his_e) ----
        # mu_z, logvar_z = self.post_head(ctx)        # [B, ah, Z]
        mu_p, logvar_p = self.prior_head(his_state_emd)  # [B, ah, Z]  # his_e 与 ctx 等长（Ts==ah）
        def post_head_on(_):  return self.post_head(ctx)
        def post_head_off(_): return (jnp.zeros_like(mu_p), jnp.zeros_like(logvar_p))
        mu_z, logvar_z = jax.lax.cond(use_human, post_head_on, post_head_off, operand=None)  # [B, ah, Z]
        
        beta_small = (kl_weight <= 1e-5)
        use_prior_z = jnp.logical_or(beta_small,jnp.logical_not(use_human))
        
        def sample_prior(_):
            eps = jax.random.normal(eps_z_rng, mu_p.shape, dtype=mu_p.dtype)
            return mu_p + jnp.exp(0.5 * logvar_p) * eps

        def sample_posterior(_):
            eps = jax.random.normal(eps_z_rng, mu_z.shape, dtype=mu_z.dtype)
            return mu_z + jnp.exp(0.5 * logvar_z) * eps

        z = jax.lax.cond(use_prior_z, sample_prior, sample_posterior, operand=None)  # [B, ah, Z]

        mask_f = his_mask.astype(mu_z.dtype)[..., None]  # [B,ah,1]
        z = mask_f * z + (1 - mask_f) * jax.random.normal(eps_pad_rng, z.shape, dtype=z.dtype)

        # ---- Decoder: (μ_noise, logσ²_noise) ----
        mu_noise, logvar_noise = self.decoder(ctx, z)    # [B, ah, D]

        # 采样噪声作为 FM 初始分布
        eps   = jax.random.normal(eps_noise_rng, mu_noise.shape, dtype=mu_noise.dtype)
        std   = jnp.exp(0.5 * logvar_noise)
        noise = mu_noise + std * eps                     # [B, ah, D]
        noise = mask_f * noise + (1 - mask_f) * jax.random.normal(eps_noise_pad_rng, noise.shape, dtype=noise.dtype)

        # ---- 采样时间 & 构造 x_t / 目标 u_t（FM）----
        B = actions.shape[0]
        time = jax.random.beta(time_rng, 1.5, 1.0, (B,)) * 0.999 + 0.001
        t = time[..., None, None]                        # [B,1,1]
        x_t = t * noise + (1.0 - t) * actions           # [B, ah, D]
        u_t = noise - actions                            # [B, ah, D]

        # ---- conditional embedding tokens ----
        cond_tokens, cond_mask = self.cond_emd(z, mu_noise, logvar_noise, rngs=cond_rng, train=train)  # [B, 1, E]

        # ---- LLM 前向（保持你原逻辑）----
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)               # [B,P,*]
        
        new_prefix_tokens = jnp.concatenate([cond_tokens, prefix_tokens], axis=1)  # [B,P+ah+1,*]
        new_prefix_mask = jnp.concatenate([cond_mask, prefix_mask], axis=1)    # [B,P+ah+1]
        
        cond_ar_mask = jnp.zeros((1,))
        new_prefix_ar_mask = jnp.concatenate([cond_ar_mask, prefix_ar_mask], axis=0)  # [B,P+ah+1]
        
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)    # [B,ah+1,*]

        input_mask = jnp.concatenate([new_prefix_mask, suffix_mask], axis=1)
        ar_mask    = jnp.concatenate([new_prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask_llm  = make_attn_mask(input_mask, ar_mask)
        positions  = jnp.cumsum(input_mask.astype(jnp.int32), axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [new_prefix_tokens, suffix_tokens],
            mask=attn_mask_llm,
            positions=positions,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])                          # [B, ah, D]

        # ---- per-step MSE（FM）----
        sigma2 = jnp.exp(logvar_noise)
        res = v_t - u_t
        nll = (res)**2 / sigma2 + jnp.log(sigma2)
        mse_step = jnp.sum(nll, axis=-1)  # [B, ah]
        step_mask = mask_f.squeeze(-1).astype(mse_step.dtype)
        mse_step = (mse_step * step_mask) / (jnp.sum(step_mask, axis=1, keepdims=True) + 1e-8)
        
        def _with_human_step(_):
            kl_step = _extra_process.kl_balanced_with_freebits(
                mu_z, logvar_z, mu_p, logvar_p,
                mask=his_mask, alpha=0.8, free_bits=0.5, reduce="none"  # [B, ah]
            )
            tcc_val = _extra_process.tcc_cycle_back_regression(
                his_state_emd, human_state_emd, mask_u=his_mask, mask_v=human_mask,
                tau=getattr(self, "tcc_tau", 0.1),
                lambda_logvar=getattr(self, "tcc_lambda_logvar", 1e-3)   # 标量
            )
            tcc_step = jnp.full_like(mse_step, tcc_val)  # 广播到 [B, ah]
            return kl_step, tcc_step

        def _without_human_step(_):
            zeros = jnp.zeros_like(mse_step)  # [B, ah]
            return zeros, zeros

        kl_step, tcc_step = jax.lax.cond(use_human, _with_human_step, _without_human_step, operand=None)
        
        def do_consistency(_):
            # g=0 / 无 human 路径：ctx0=his_state_emd，z0 用 prior，cond0 基于 (z0, mu0, logvar0)
            eps_z0 = jax.random.normal(eps_z_rng, mu_p.shape, dtype=mu_p.dtype)  # 共享 rng 以降方差（简化）
            z0 = mu_p + jnp.exp(0.5 * logvar_p) * eps_z0
            mu0, logv0 = self.decoder(his_state_emd, z0)

            # 用“相同的 x_t（主分支的）”做 suffix，保证比较同一点的速度场
            cond0, mask0 = self.cond_emd(z0, mu0, logv0, rngs=cond_rng, train=train)

            new_pref_tok0  = jnp.concatenate([cond0, prefix_tokens], axis=1)
            new_pref_mask0 = jnp.concatenate([mask0,  prefix_mask],  axis=1)
            new_pref_ar0   = jnp.concatenate([jnp.zeros((1, )), prefix_ar_mask], axis=0)

            in_mask0 = jnp.concatenate([new_pref_mask0, suffix_mask], axis=1)
            ar_mask0 = jnp.concatenate([new_pref_ar0,   suffix_ar_mask], axis=0)
            attn0    = make_attn_mask(in_mask0, ar_mask0)
            pos0     = jnp.cumsum(in_mask0.astype(jnp.int32), axis=1) - 1

            (_, suffix_out0), _ = self.PaliGemma.llm(
                [new_pref_tok0, suffix_tokens],  # suffix_tokens 复用主分支（同 x_t, t）
                mask=attn0,
                positions=pos0,
            )
            v_t0 = self.action_out_proj(suffix_out0[:, -self.action_horizon:])  # [B,ah,D]

            # 一致性损失：按 (1 - g) 加权（g 大时少约束，g 小时逼近 no-human）
            w_cons = (1.0 - gate_g).astype(v_t.dtype)            # [B,1,1]
            w_cons = jnp.broadcast_to(w_cons, v_t.shape)         # [B,ah,D]
            cons = jnp.mean(((v_t - v_t0) ** 2) * w_cons, axis=-1)  # [B,ah]
            cons = (cons * step_mask) / (jnp.sum(step_mask, axis=1, keepdims=True) + 1e-8)
            return cons
        
        do_cons = train & (jax.random.uniform(drop_rng, ()) < self.consistency_prob)
        cons_step = jax.lax.cond(do_cons, do_consistency, lambda _: jnp.zeros_like(mse_step), operand=None)

        loss_step = mse_step + kl_weight * kl_step + tcc_weight * tcc_step + reg_gate +  self.consistency_lambda * cons_step
        return {"total": loss_step, "kl": kl_step, "tcc": tcc_step, "kl_weight": kl_weight, "tcc_weight": tcc_weight, "reg_gate": reg_gate, "consistency": cons_step}  # [B, ah]

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions: # (batch_size, action_horizon, action_dim)
        
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        # dt = -0.1
        dt = -1.0 / num_steps  # from t=1 to t=0 each step = dt
        batch_size = observation.state.shape[0]
        # TODO
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2 # 

        x_0, _ = jax.lax.while_loop(cond, # 循环条件
                                    step, 
                                    (noise, 1.0))
        return x_0
     
    @override
    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,    # 随机数生成器的键数组
        observation: _model.Observation,  
        prefix_actions: jax.Array, 
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule,
        max_guidance_weight: float,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        
        # Add a batch dim to prefix_actions # e.g. (30, 14) -> (1, 30, 32)
        prefix_actions = prefix_actions[None, ...]
        prefix_actions = jnp.concatenate([prefix_actions, jnp.zeros((batch_size, self.action_horizon, self.action_dim-prefix_actions.shape[-1]))], axis=2)


        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def get_prefix_weights(start: int, end: int, total: int, schedule) -> jax.Array:
            """With start=2, end=6, total=10, the output will be:
            1  1  4/5 3/5 2/5 1/5 0  0  0  0
                ^              ^
                start           end
            `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
            paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
            entire prefix is attended to.

            `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
            if `end` is 0, then the entire prefix will always be ignored.
            """
            start = jnp.minimum(start, end)
            if schedule == "ones":
                w = jnp.ones(total)
            elif schedule == "zeros":
                w = (jnp.arange(total) < start).astype(jnp.float32)
            elif schedule == "linear" or schedule == "exp":
                w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
                if schedule == "exp":
                    w = w * jnp.expm1(w) / (jnp.e - 1)
            else:
                raise ValueError(f"Invalid schedule: {schedule}")
            return jnp.where(jnp.arange(total) >= end, 0, w)

        def pinv_corrected_velocity(
            v_t_fn, # ([ah ad], float) -> [ah ad]
            x_t: jax.Array, # [b ah ad]
            t: float,
            prefix_actions: jax.Array, # [b ah ad]
            inference_delay: int,
            prefix_attention_horizon: int,
            max_guidance_weight: float,
        ) -> jax.Array: # [b ah ad]
            @jax.vmap
            def _pinv_corrected_velocity(
                x_t: jax.Array, # [ah ad]
                y: jax.Array, # [ah ad]
            ) -> jax.Array: # [ah ad]
                def denoiser(x_t: jax.Array) -> tuple[jax.Array, jax.Array]: # [ah ad]
                    v_t = v_t_fn(x_t, t)
                    return x_t - v_t * t, v_t

                x_0, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
                # print(f"x_0: {x_0.shape}, v_t: {v_t.shape}")
                # print(f"x_0: {x_0[:][0]}, y: {y[:][0]}")
                # jax.debug.print("x_0 shape = {x}", x=x_0.shape)
                # jax.debug.print("y shape = {y}", y=y.shape)
                # jax.debug.print("x_0[0] = {x}", x=x_0[0])
                # jax.debug.print("y[0] = {y}", y=y[0])
                error = (y - x_0) * get_prefix_weights(inference_delay, prefix_attention_horizon, prefix_actions.shape[1], 'exp')[:, None]
                # jax.debug.print("error mean = {x}", x=jnp.mean(jnp.abs(error)))
                pinv_correction = vjp_fun(error)[0]
                # constants from paper
                inv_r2 = (t**2 + (1 - t) ** 2) / (t**2)
                c = jnp.nan_to_num(t / (1 - t), posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                return v_t - guidance_weight * pinv_correction

            return _pinv_corrected_velocity(x_t, prefix_actions)

        def v_t_step(
                x_t: jax.Array, # [ah ad]
                time: jax.Array, # []
                ):
            # TODO: find better way to support jax.vmap
            x_t = x_t[None, ...]
            time = time[None, ...]

            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, time
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return v_t[0, ...] # TODO: remove this since it's not super vectorized

        def rtc_step(carry):
            x_t, time = carry
            guided_vt = pinv_corrected_velocity(v_t_step, x_t, time, prefix_actions, inference_delay, prefix_attention_horizon, max_guidance_weight)
            return x_t + dt * guided_vt, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, rtc_step, (noise, 1.0))
        # jax.debug.print("x_0, joint 0 = {x}", x=x_0[0][0])
        # jax.debug.print(
        #     "x[{b}, :, {j}] = {x}",
        #     b=0,
        #     j=0,
        #     x=x_0[0, :, 0]  
        # )
        # jax.debug.print(
        #     "pre[{b}, :, {j}] = {x}",
        #     b=0,
        #     j=0,
        #     x=prefix_actions[0, :, 0]          # 关键切片：batch 固定、时间全取、关节固定
        # )
        # jax.debug.print("previx_actions, joint 0 = {x}", x=prefix_actions[0][0])
        return x_0
     
    @override
    def sample_actions_human(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        human_action: dict,
        his_state: dict,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:  # [B, ah, D]
        assert num_steps > 0
        dt = -1.0 / num_steps           # 从 t=1 积分到 0
        # 预处理
        observation = _model.preprocess_observation(None, observation, train=False)

        # ---- 取数据 ----
        human_action_data = human_action["data"]          # [B, Th, D_in]
        his_state_data    = his_state["data"]             # [B, Ts, D_in]
        human_mask        = human_action.get("mask", jnp.ones(human_action_data.shape[:2], bool))
        his_mask          = his_state.get("mask",   jnp.ones(his_state_data.shape[:2], bool))
        state_start       = his_state["curr_id"]

        human_mask = human_mask.astype(bool)
        his_mask   = his_mask.astype(bool)
        B = his_state_data.shape[0]
        batch_size = B

        his_window_size = 50
        his_state_data, his_mask = _extra_process.get_history_window(
            his_state_data, his_mask, state_start, his_window_size
        )

        # ---- 编码到 E ----
        his_state_emd = self.state_enc(his_state_data, his_mask, pos_start=0, deterministic=True)  # [B, Ts, E]

        # ---- prior / posterior 路径 ----
        cond = jnp.any(human_mask)  # batch 内只要有 human 就走 posterior
        def infer_without_human(_):
            mu_p, logvar_p = self.prior_head(his_state_emd)               # [B, ah, Z]
            eps = jax.random.normal(rng, mu_p.shape, dtype=mu_p.dtype)
            z = mu_p + jnp.exp(0.5 * logvar_p) * eps
            mu_noise, logvar_noise = self.decoder(his_state_emd, z)       # ctx=his_state_emd
            return z, mu_noise, logvar_noise

        def infer_with_human(_):
            human_state_emd = self.state_enc(human_action_data, human_mask, pos_start=0, deterministic=True)  # [B, Th, E]
            ctx, _ = self.cross_enc(
                his_state_emd, human_state_emd,
                kv_mask=human_mask, q_mask=his_mask,
                q_pos_start=0, kv_pos_start=0, deterministic=True
            )                                                            # ctx: [B, Ts(=ah), E]
            mu_z, logvar_z = self.post_head(ctx)                         # [B, ah, Z]
            eps = jax.random.normal(rng, mu_z.shape, dtype=mu_z.dtype)
            z = mu_z + jnp.exp(0.5 * logvar_z) * eps
            mu_noise, logvar_noise = self.decoder(ctx, z)
            return z, mu_noise, logvar_noise

        z, mu_noise, logvar_noise = jax.lax.cond(cond, infer_with_human, infer_without_human, operand=None)

        # ---- 采样噪声作为 x(1) ----
        rng, rng_eps_noise = jax.random.split(rng)
        eps  = jax.random.normal(rng_eps_noise, mu_noise.shape, dtype=mu_noise.dtype)
        std  = jnp.exp(0.5 * logvar_noise)
        noise = mu_noise + std * eps                                      # x(1) [B, ah, D]

        # ================== 构建 [COND | PREFIX] 并建立 KV cache ==================
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)     # [B,P,E], [B,P], [B,P]
        # 用与训练一致的 cond_embed（time 这里取常数 1.0 即可；它只作为条件上下文）

        cond_tokens, cond_mask = self.cond_emd(z, mu_noise, logvar_noise, train=False)   # [B,K,E], [B,K]
        cond_ar_mask = jnp.zeros_like(cond_mask)

        # 把条件放最前（[COND | PREFIX]），并按 token 维 axis=1 拼接 AR mask
        new_prefix_tokens   = jnp.concatenate([cond_tokens, prefix_tokens], axis=1)     # [B,K+P,E]
        new_prefix_mask     = jnp.concatenate([cond_mask,   prefix_mask],   axis=1)     # [B,K+P]
        new_prefix_ar_mask  = jnp.concatenate([cond_ar_mask, prefix_ar_mask], axis=1)   # [B,K+P]

        prefix_attn_mask = make_attn_mask(new_prefix_mask, new_prefix_ar_mask)          # [B,K+P,K+P]
        positions_prefix = jnp.cumsum(new_prefix_mask.astype(jnp.int32), axis=1) - 1    # [B,K+P]

        # 只跑 prefix，建立 KV cache
        _, kv_cache = self.PaliGemma.llm([new_prefix_tokens, None], mask=prefix_attn_mask, positions=positions_prefix)

        # ================== FM 推理积分：从 t=1 → 0 ==================
        def step_fn(carry, i):
            x_t = carry                                 # [B, ah, D]
            # 当前时间（均匀 Euler）
            t_now = 1.0 + (i * dt)                      # i=0 -> t=1.0
            time_vec = jnp.full((batch_size,), t_now, dtype=jnp.float32)

            # suffix 编码
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time_vec)  # [B,S,E], [B,S], [B,S]

            # 构造 suffix 的注意力掩码（让其可见到 [COND|PREFIX] 与自身的因果结构）
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)                              # [B,S,S]
            prefix_to_suffix = jnp.broadcast_to(new_prefix_mask[:, None, :], (batch_size, suffix_tokens.shape[1], new_prefix_mask.shape[1]))  # [B,S,K+P]
            full_attn_mask   = jnp.concatenate([prefix_to_suffix, suffix_attn_mask], axis=-1)          # [B,S,K+P+S]

            # positions：suffix 紧接在 prefix 后
            positions_suffix = jnp.sum(new_prefix_mask, axis=-1, dtype=jnp.int32)[:, None] + \
                            jnp.cumsum(suffix_mask.astype(jnp.int32), axis=-1) - 1

            # 只前向 suffix（复用 KV cache）
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])   # [B, ah, D]

            x_next = x_t + dt * v_t
            return x_next, None

        # 用 fori_loop 更稳（避免 while 的浮点边界）
        x_1 = noise
        x_T, _ = jax.lax.fori_loop(0, num_steps, step_fn, x_1)

        return x_T  # [B, ah, D]
