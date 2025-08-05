from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import aloha_policy
from tqdm import tqdm  # 导入 tqdm 进度条

# 获取配置
config = config.get_config("pi0_piper_pen_uncap_low_mem_finetune")
checkpoint_dir = "/home/research1/project/openpi/checkpoints/pi0_piper_pen_uncap_low_mem_finetune/my_experiment20250717/19999"

# 训练政策的创建（假设这个过程需要一些时间）
print("Loading trained policy...")

# 模拟加载过程
policy = policy_config.create_trained_policy(config, checkpoint_dir)


# 创建一个 Aloha 示例
print("Creating example for inference...")

example = aloha_policy.make_aloha_example()

# 运行推理
print("Running inference...")
action_chunk = policy.infer(example)["actions"]

# 输出结果
print("Inference completed.")
print(action_chunk)

print(action_chunk.shape)
