import dataclasses

import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

config = _config.get_config("pi0_franka_low_mem_finetune")
checkpoint_dir = "/home/io/ld/openpi/checkpoints/pi0_franka_low_mem_finetune/delta_model/39999"

policy = _policy_config.create_trained_policy(config, checkpoint_dir)
# model = config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

key = jax.random.key(0)
config = dataclasses.replace(config, batch_size=3)

loader = _data_loader.create_data_loader(config, num_batches=500)
data_iter = iter(loader)

for i in range(500):
    obs, act = next(data_iter)
    result, groud = policy.infer_dataset(obs, act)
    print(result['actions'], groud['actions'])