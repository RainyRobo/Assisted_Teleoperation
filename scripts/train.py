import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

def linear_warmup(step: jnp.ndarray, warmup_steps: int, max_value: float) -> jnp.ndarray:
    if warmup_steps <= 0:
        return jnp.array(max_value, dtype=jnp.float32)
    r = jnp.clip(step / float(warmup_steps), 0.0, 1.0)
    return jnp.array(max_value, dtype=jnp.float32) * r

def cosine_warmup(step: jnp.ndarray, warmup_steps: int, max_value: float) -> jnp.ndarray:
    if warmup_steps <= 0:
        return jnp.array(max_value, dtype=jnp.float32)
    r = jnp.clip(step / float(warmup_steps), 0.0, 1.0)
    return jnp.array(max_value, dtype=jnp.float32) * 0.5 * (1.0 - jnp.cos(jnp.pi * r))

def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    # 创建优化器
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


from flax import traverse_util
def make_pred(includes, excludes=()):
    def pred(path):
        p = "/".join(path) if isinstance(path, tuple) else path
        return any(inc in p for inc in includes) and not any(exc in p for exc in excludes)
    return pred

module_pred = make_pred(
    includes=("state_enc", "cross_enc", "decoder", "post_head", "prior_head", "cond_emd", "cross_gate"),
    excludes=("bias", "scale", "pos_embedding", "input_embedding"),
)

def grad_norm_filtered(grads, module_pred=None):
    """支持对 grads 做按路径筛选后的 L2 范数；若筛不到就回退到全局范数。"""
    try:
        if module_pred is None:
            return optax.global_norm(grads)
        flat = traverse_util.flatten_dict(grads, sep="/")
        sel = {k: v for k, v in flat.items() if module_pred(k)}
        if not sel:
            return optax.global_norm(grads)
        sub = traverse_util.unflatten_dict(sel)
        return optax.global_norm(sub)
    except Exception:
        return optax.global_norm(grads)

# -------- 阶梯式 warmup ----------
def step_quantize(step, N):
    return (step // N) * N

def cosine_warmup_stair(step, warmup_steps, max_val, N):
    s = jnp.clip(step_quantize(step, N) / warmup_steps, 0., 1.)
    return max_val * 0.5 * (1. - jnp.cos(jnp.pi * s))


@at.typecheck
def train_step(
    config: _config.TrainConfig, 
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, dict, dict],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """单步训练函数：前向、loss、反向传播、优化器更新、日志收集"""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    
    step_scalar = jnp.array(state.step, dtype=jnp.float32)
    beta_kl = cosine_warmup(step_scalar, 2_000, 1e-3)    # KL 权重
    w_tcc_raw = cosine_warmup(step_scalar, 2_000, 1e-3)  # TCC 权重
    w_tcc = jnp.clip(w_tcc_raw, 0., None)
    
    # === 定义损失函数 ===
    @at.typecheck
    def loss_fn_extra(
        model: _model.BaseModel, 
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        human_action: dict,
        his_state: dict,
    ):
        # compute_loss_extra 已经返回一个 dict
        loss_dict = model.compute_loss_extra(
            rng,
            observation,
            actions,
            human_action,
            his_state,
            train=True,
            kl_weight=beta_kl,
            tcc_weight=w_tcc,
        )
        # 主损失 = total 的 batch 均值
        return jnp.mean(loss_dict["total"]), loss_dict

    # === 前向 + 梯度 ===
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, human_action, his_state = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_dict), grads = nnx.value_and_grad(
        loss_fn_extra, argnums=diff_state, has_aux=True
    )(model, train_rng, observation, actions, human_action, his_state)

    # === 参数更新 ===
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )

    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # === 参数范数监控 ===
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    # === Wandb-friendly info dict ===
    info = {
        # 主损失
        "loss/total": jnp.mean(loss_dict["total"]),

        # KL / TCC
        "loss/kl": jnp.mean(loss_dict["kl"]),
        "weight/kl": loss_dict["kl_weight"],

        # 优化器监控
        "opt/grad_norm": optax.global_norm(grads),
        "opt/param_norm": optax.global_norm(kernel_params),
    }

    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 分布式训练
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding, #分片
        shuffle=True, #打乱
    )

    # 数据迭代器：
    data_iter = iter(data_loader)
    batch = next(data_iter)

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # 初始化训练
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    # logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
