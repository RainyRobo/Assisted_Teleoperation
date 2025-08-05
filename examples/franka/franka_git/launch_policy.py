import logging
import socket

from openpi.policies import policy_config as _policy_config  # policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config  # traing_config

# 定义模型的ckpt
train_config_name = "pi0_franka_low_mem_finetune"
model_name = "latest_model"
checkpoint_id = 39999

# 创建policy
config = _config.get_config(train_config_name)
policy = _policy_config.create_trained_policy(
    config,
    f"/home/io/ld/openpi/checkpoints/{train_config_name}/{model_name}/{checkpoint_id}",
)
print("load model success!")
policy_metadata = policy.metadata

# 创建 server
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

server = websocket_policy_server.WebsocketPolicyServer(
    policy=policy,
    host="0.0.0.0",
    port=8000,
    metadata=policy_metadata,
)
server.serve_forever()
