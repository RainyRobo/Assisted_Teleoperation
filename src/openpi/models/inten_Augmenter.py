import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

# ==============================================================================
#           数据增强类 (支持随机化强度和范围) + mask / 可选 padding
# ==============================================================================

class TrajectoryAugmenter:
    """
    一个封装了轨迹数据增强逻辑的类。

    该类在初始化时接收增强参数的范围，并在每次被调用时，
    从该范围中随机采样一个具体值来执行增强。
    现在支持:
      - 返回 mask（有效步长为1，pad为0）
      - 可选将增强后轨迹 pad 到指定长度（如原始长度）
    """
    def __init__(
        self,
        enable_augmentation: bool = True,
        noise_strength_range: Union[float, Tuple[float, float]] = (0.0, 0.05),
        crop_percentage_range: Union[float, Tuple[float, float]] = (0.0, 0.5),
        pad_value: float = 0.0,  # 连续值轨迹的padding值
    ):
        """
        初始化增强器。

        Args:
            enable_augmentation (bool): 是否启用数据增强的总开关。
            noise_strength_range (Union[float, Tuple[float, float]]): 
                相对噪声强度的范围。每次调用将从此范围中均匀采样。
                如果提供单个浮点数，则每次使用该固定值。
            crop_percentage_range (Union[float, Tuple[float, float]]): 
                最大裁剪比例的范围。每次调用将从此范围中均匀采样一个
                最大比例，然后进行0到该比例的随机裁剪。
            pad_value (float): 当需要pad时用于填充值（连续值建议0.0）。
        """
        self.enable_augmentation = enable_augmentation

        # 处理噪声强度参数
        if isinstance(noise_strength_range, (int, float)):
            self.noise_strength_range = (float(noise_strength_range), float(noise_strength_range))
        else:
            self.noise_strength_range = (float(noise_strength_range[0]), float(noise_strength_range[1]))
        
        # 处理裁剪比例参数
        if isinstance(crop_percentage_range, (int, float)):
            self.crop_percentage_range = (float(crop_percentage_range), float(crop_percentage_range))
        else:
            self.crop_percentage_range = (float(crop_percentage_range[0]), float(crop_percentage_range[1]))

        self.pad_value = float(pad_value)

        # 参数验证
        assert 0.0 <= self.noise_strength_range[0] <= self.noise_strength_range[1], "噪声强度范围非法"
        assert 0.0 <= self.crop_percentage_range[0] <= self.crop_percentage_range[1] <= 1.0, "裁剪比例范围非法"

    def _add_relative_gaussian_noise(self, trajectory: np.ndarray, strength: float) -> np.ndarray:
        """[内部方法] 根据给定的强度值添加相对高斯噪声。"""
        if strength == 0:
            return trajectory
            
        data_std = np.std(trajectory, axis=0)
        data_std = np.where(data_std == 0, 1e-6, data_std)
        
        noise_std_per_dim = data_std * strength
        noise = np.random.normal(loc=0.0, scale=noise_std_per_dim, size=trajectory.shape)
        
        return trajectory + noise

    def _random_crop_from_end(self, trajectory: np.ndarray, max_percentage: float) -> np.ndarray:
        """[内部方法] 根据给定的最大比例进行末尾裁剪。"""
        if max_percentage == 0:
            return trajectory
            
        original_len = trajectory.shape[0]
        max_steps_to_crop = int(original_len * max_percentage)
        steps_to_crop = np.random.randint(0, max_steps_to_crop + 1)
        
        if steps_to_crop == 0:
            return trajectory
            
        new_len = original_len - steps_to_crop
        return trajectory[:new_len]

    def _maybe_pad_with_mask(
        self,
        traj: np.ndarray,
        target_len: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 traj 右侧pad到 target_len，并返回 (padded_traj, mask)。
        若 target_len 为 None 或 <= 当前长度，则不pad，mask 全1。
        mask 形状为 (T,)；traj 为 (T, D)。
        """
        cur_len = traj.shape[0]
        mask = np.ones((cur_len,), dtype=bool)

        if target_len is None or target_len <= cur_len:
            return traj, mask

        pad_steps = target_len - cur_len
        D = traj.shape[1]
        pad_block = np.full((pad_steps, D), self.pad_value, dtype=traj.dtype)
        padded_traj = np.concatenate([traj, pad_block], axis=0)

        padded_mask = np.zeros((target_len,), dtype=bool)
        padded_mask[:cur_len] = True
        return padded_traj, padded_mask

    def __call__(
        self,
        ep_state: np.ndarray,
        return_mask: bool = False,
        pad_to: Optional[Union[int, str]] = None,
    ):
        """
        对输入的轨迹执行随机化的增强操作。

        Args:
            ep_state (np.ndarray): 形状 (T, D) 的轨迹。
            return_mask (bool): 是否返回mask（有效为1 / True，pad为0 / False）。
            pad_to (Optional[Union[int, str]]):
                - None: 不做pad（默认）
                - 'original': pad回原始长度 ep_state.shape[0]
                - int: pad到给定长度

        Returns:
            若 return_mask=False: np.ndarray  (增强后的轨迹；若 pad_to 指定则为pad后的)
            若 return_mask=True:  (np.ndarray, np.ndarray, int)
                分别为 (轨迹, mask, valid_len)，其中 valid_len 是裁剪后的真实长度。
        """
        assert ep_state.ndim == 2, "ep_state 需要是二维 (T, D)"
        if not self.enable_augmentation:
            if return_mask:
                target_len = ep_state.shape[0] if pad_to == 'original' else (pad_to if isinstance(pad_to, int) else None)
                out_traj, mask = self._maybe_pad_with_mask(ep_state, target_len)
                return out_traj, mask, ep_state.shape[0]
            return ep_state

        # === 随机采样参数 ===
        current_noise_strength = np.random.uniform(*self.noise_strength_range)
        current_max_crop_percentage = np.random.uniform(*self.crop_percentage_range)

        # === 应用增强 ===
        cropped_traj = self._random_crop_from_end(
            ep_state, max_percentage=current_max_crop_percentage
        )
        valid_len = cropped_traj.shape[0]

        augmented_traj = self._add_relative_gaussian_noise(
            cropped_traj, strength=current_noise_strength
        )

        # === 可选 pad + mask ===
        target_len = None
        if pad_to == 'original':
            target_len = ep_state.shape[0]
        elif isinstance(pad_to, int):
            target_len = pad_to

        if return_mask:
            out_traj, mask = self._maybe_pad_with_mask(augmented_traj, target_len)
            return out_traj, mask, valid_len
        else:
            if target_len is not None:
                out_traj, _ = self._maybe_pad_with_mask(augmented_traj, target_len)
                return out_traj
            return augmented_traj

    # ------------------ 修改后的可视化静态方法 ------------------
    @staticmethod
    def visualize(
        original_trajectory: np.ndarray,
        augmented_trajectory: np.ndarray,
        dims_to_plot: Optional[List[int]] = None,
        title: str = "Trajectory Augmentation Comparison",
        mask: Optional[np.ndarray] = None,  # 可选mask用于高亮有效区域
        save_path: str = "trajectory_aug.png",  # 新增：保存路径
        dpi: int = 150,
    ):
        """
        可视化单个原始轨迹和单个增强后轨迹的对比图。
        若传入 mask（形状 (T,)），会用浅灰色背景标注 pad 区域（mask==0 的区间）。
        图像保存到 save_path，而不是 show。
        """
        assert original_trajectory.ndim == 2 and augmented_trajectory.ndim == 2, "轨迹应为 (T, D)"
        if dims_to_plot is None:
            dims_to_plot = list(range(min(3, original_trajectory.shape[1])))

        num_dims = len(dims_to_plot)
        fig, axes = plt.subplots(num_dims, 1, figsize=(14, num_dims * 3), sharex=True)
        if num_dims == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16)

        original_time = np.arange(original_trajectory.shape[0])
        augmented_time = np.arange(augmented_trajectory.shape[0])

        # 若提供了mask，标注padding区域（mask==0）
        if mask is not None:
            assert mask.ndim == 1 and mask.shape[0] == augmented_trajectory.shape[0], "mask 形状应为 (T,) 且与增强轨迹长度一致"
            padded_idxs = np.where(~mask)[0]
        else:
            padded_idxs = np.array([], dtype=int)

        for i, dim_idx in enumerate(dims_to_plot):
            ax = axes[i]

            ax.plot(
                original_time,
                original_trajectory[:, dim_idx],
                label=f'Original (Dim {dim_idx})',
                linestyle='-',
                linewidth=2,
            )
            ax.plot(
                augmented_time,
                augmented_trajectory[:, dim_idx],
                label=f'Augmented (Dim {dim_idx})',
                linestyle='--',
            )

            if padded_idxs.size > 0:
                gaps = np.where(np.diff(padded_idxs) > 1)[0]
                starts = np.r_[padded_idxs[0], padded_idxs[gaps + 1]]
                ends = np.r_[padded_idxs[gaps], padded_idxs[-1]]
                for s, e in zip(starts, ends):
                    ax.axvspan(s, e + 1, alpha=0.15)

            ax.legend()
            ax.set_ylabel('State Value')
            ax.grid(True, which='both', linestyle=':', linewidth=0.7)

        axes[-1].set_xlabel('Time Step')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # 保存而不是 show
        plt.savefig(save_path, dpi=dpi)
        plt.close(fig)
