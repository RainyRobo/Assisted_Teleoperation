import fnmatch
import json
import os
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm

REPO_NAME = "dian/franka_stap_cup_0620"


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                imgs_array.append(
                    cv2.imdecode(data, cv2.IMREAD_COLOR)
                )  # [H, W, C] = [480, 640, 3]，此时是 RGB 格式的！！！
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array

    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with h5py.File(ep_path, "r") as ep:
        state = ep["/observations/qpos"][:]
        action = ep["/action"][:]
        imgs_per_cam = None
        imgs_per_cam = load_raw_images_per_camera(
            ep,
            ["cam_high", "cam_left"],
        )

    return imgs_per_cam, state, action


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "joints": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joints"],
            },
            "gripper": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    hdf5_files = []
    for root, _, files in os.walk(data_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)

    episodes = range(len(hdf5_files))

    max = 0
    min = 999999
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action = load_raw_episode_data(ep_path)
        state[:, 7] = 1.0 - state[:, 7]
        action[:, 7] = 1.0 - action[:, 7]  # 1.0 代表闭合， 0.0代表张开
        num_frames = state.shape[0]

        if num_frames > max:
            max = num_frames
        if num_frames < min:
            min = num_frames
        # add prompt
        dir_path = os.path.dirname(ep_path)
        json_Path = f"{dir_path}/instructions.json"

        with open(json_Path) as f_instr:
            instruction_dict = json.load(f_instr)
            instructions = instruction_dict["instructions"]
            instruction = np.random.choice(instructions)

        for i in range(num_frames):
            dataset.add_frame(
                {
                    "image": imgs_per_cam["cam_left"][i],
                    "wrist_image": imgs_per_cam["cam_high"][i],
                    "joints": state[i, :7],
                    "gripper": np.array([state[i, 7]], dtype=np.float32),
                    "actions": action[i],
                }
            )
        dataset.save_episode(task=instruction)

        # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)
    print(max, min)  # max = 491 min =221 
    # max 899 min 242 total frame 145034


main("/home/io/ld/data/hdf5_0619")
