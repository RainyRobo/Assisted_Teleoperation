from __future__ import annotations
import numpy as np
import torch
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def get_episode_states(dataset: LeRobotDataset, episode_index: int) -> torch.Tensor:
    """
    根据 episode_index 读取一整集的 observation.state
    返回形状 [T, D] 的 float32 张量
    
    参数:
        dataset: 已初始化的 LeRobotDataset 对象
        episode_index: int, 目标 episode 的编号 (从 0 开始)
    """
    hf = dataset.hf_dataset

    # 确认列存在
    required_cols = ["episode_index", "observation.state"]
    for col in required_cols:
        if col not in hf.column_names:
            raise KeyError(f"数据集中缺少 '{col}' 列, 实际列有: {hf.column_names}")

    # 过滤指定 episode
    hf_ep = hf.filter(lambda ex: ex["episode_index"] == episode_index)
    if len(hf_ep) == 0:
        raise ValueError(f"未找到 episode_index={episode_index} 的数据")

    # 转 pandas 再堆叠
    df = hf_ep.to_pandas()
    states_list = [np.asarray(s, dtype=np.float32) for s in df["observation.state"].tolist()]
    states = torch.from_numpy(np.stack(states_list, axis=0))

    return states


# # ---------------- 示例 ----------------
# if __name__ == "__main__":
#     # 加载某个数据集
#     ds = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")   # 或本地路径
    
#     # 获取第 0 集的 states
#     states = get_episode_states(ds, 0)
#     print("states.shape =", states.shape)
#     print(states[:3])  # 打印前三步
