

from __future__ import annotations
import numpy as np
import torch
import pandas as pd

# 1) 载入数据集：既可用 Hub repo_id，也可用本地根目录路径
#    例如 "lerobot/pusht"、"lerobot/xarm_lift_medium" 或你自己的 ${USER}/my_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

repo_or_path = "lerobot/aloha_sim_transfer_cube_human"   # ← 换成你的数据集
ds = LeRobotDataset(repo_or_path)

# 2) 选择 episode 索引（从 0 开始）
episode_id = 0  # ← 想读第几集就填几

# 3) 用 hf_dataset 里的 episode_index 列筛选整集帧
#    这样能一次性拿到该集所有时间步的数据（更快更干净）
hf = ds.hf_dataset  # datasets.Dataset (Arrow/Parquet 后端)

# 安全检查：确保列存在（有些数据集可能没有图像列，但 state/episode_index 通常都有）
required_cols = ["episode_index", "observation.state"]
missing = [c for c in required_cols if c not in hf.column_names]
if missing:
    raise KeyError(f"数据集中缺少所需列: {missing}\n当前列: {hf.column_names}")

# 过滤出本 episode 的所有帧；再转 pandas 方便批量堆叠
hf_ep = hf.filter(lambda ex: ex["episode_index"] == episode_id)
df = hf_ep.to_pandas()  # 每行是一帧

if len(df) == 0:
    raise ValueError(f"episode_index={episode_id} 未找到帧数据（可能越界或数据损坏）。")

# 4) 将 observation.state 列的逐帧向量堆叠为 T×D 的张量
#    state 通常是 list/ndarray；统一转成 float32
states_list = [np.asarray(s, dtype=np.float32) for s in df["observation.state"].tolist()]
states = torch.from_numpy(np.stack(states_list, axis=0))  # [T, D]

print(f"Episode {episode_id}: states shape = {tuple(states.shape)}  (T×D)")
print("示例前3步：\n", states[:3])
