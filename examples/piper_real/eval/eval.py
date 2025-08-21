import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import aloha_policy as _aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


key = jax.random.key(0)
config = _config.get_config("pi0_piper_pen_uncap_low_mem_finetune")
config = dataclasses.replace(config, batch_size=2)

# print(config)

checkpoint_dir = "/home/research1/project/openpi/checkpoints/pi0_piper_pen_uncap_low_mem_finetune/my_experiment20250717/5000"

# Create a model from the checkpoint.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)


loader = _data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=True)
data_iter = iter(loader)

for i in range(20):
    obs, act = next(data_iter)
    
    print(obs)
    result, groud = policy.infer(obs, act)
    print(result['actions'], groud['actions'])