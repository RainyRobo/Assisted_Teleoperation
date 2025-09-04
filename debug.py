
from src.openpi.models.inten_Augmenter import TrajectoryAugmenter
import numpy as np
# --- 1. 使用范围来初始化增强器 ---
aug = TrajectoryAugmenter(
    enable_augmentation=True,
    noise_strength_range=(0.0, 0.5),
    crop_percentage_range=(0.0, 0.5),
    pad_value=0.0,
)

traj = np.random.randn(120, 7)  # (T=120, D=7)

# 1) 保持变长：只要增强后的轨迹（无mask、无padding）
aug_traj = aug(traj, return_mask=False)

# 2) 要mask但不pad：返回 (traj, mask, valid_len)，mask 全1
aug_traj2, mask2, valid_len2 = aug(traj, return_mask=True)

# 3) 要mask并pad回原始长度：返回 (padded_traj, mask, valid_len)
aug_traj3, mask3, valid_len3 = aug(traj, return_mask=True, pad_to='original')

# 4) 可视化（带mask的padding区域高亮）
TrajectoryAugmenter.visualize(traj, aug_traj3, dims_to_plot=[0,1,2], mask=mask3)
