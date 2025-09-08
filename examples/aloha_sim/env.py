import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class AlohaSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation."""

    def __init__(self, task: str, obs_type: str = "pixels_agent_pos", seed: int = 0) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        self._gym = gymnasium.make(task, obs_type=obs_type, max_episode_steps=600)

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0
        
        self.history = []  # 用来记录每个时间步骤的 (time, action_dim) 数组
        # 确保给定文件路径和文件名
        self.history_file = "/home/liuyu/project/Assisted_Teleoperation/robot_history.npy"

    @override
    def reset(self) -> None:
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False
        self._episode_reward = 0.0

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    @override
    def apply_action(self, action: dict) -> None:
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = terminated or truncated
        self._episode_reward = max(self._episode_reward, reward)
        
        
        # # 记录当前时间步骤、动作维度
        # current_time = len(self.history)  # 当前时间步是历史记录的长度
        # action_dim = len(action["actions"])  # 获取动作的维度
        
        # # 将时间步、动作维度保存到历史记录中
        self.history.append(action["actions"])

    def _convert_observation(self, gym_obs: dict) -> dict:
        img = gym_obs["pixels"]["top"]
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        # Convert axis order from [H, W, C] --> [C, H, W]
        img = np.transpose(img, (2, 0, 1))

        return {
            "state": gym_obs["agent_pos"],
            "images": {"cam_high": img},
        }

    def get_history(self):
        """
        返回存储的 (time, action_dim) 数组，并存储到本地
        """
        history_array = np.array(self.history)
        # 确保保存到文件的路径是有效的
        np.save(self.history_file, history_array)  # 将历史记录保存到本地文件
        print(f"History saved to {self.history_file}")  # 输出确认消息
