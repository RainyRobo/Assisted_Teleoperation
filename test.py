from waypoint_extraction.extract_waypoints import dp_waypoint_selection, greedy_waypoint_selection, backtrack_waypoint_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os




DATA_DIR = f"~/.cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human/data/chunk-000"
EPI_TEMPL = """episode_{num:06d}.parquet"""

num = 0
data_path = os.path.expanduser(os.path.join(DATA_DIR, EPI_TEMPL.format(num=num)))
print(f"Loading data from {data_path}")

df = pd.read_parquet(data_path)

# states = df[[col for col in df.columns if "state" in col]].to_numpy()
states = df["observation.state"].to_numpy()
left_arm_qpos = np.stack(states).squeeze()[:, :6]
right_arm_qpos = np.stack(states).squeeze()[:, 7:13]
all_data = np.concatenate([left_arm_qpos, right_arm_qpos], axis=1)

print(all_data.shape)

err_threshold = 0.01

waypoints = greedy_waypoint_selection(env=None,
                actions=all_data,
                gt_states=all_data,
                err_threshold=err_threshold,
                pos_only=True,)

print(waypoints)