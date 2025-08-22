import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
            "low_0_rgb": "cam_low" #新增
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


GRIPPER_IDXS = (6, 13)

def _as2d(x: np.ndarray):
    """Return (arr2d, batched_flag). Accepts rank-1 or rank-2 only."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :], False
    if x.ndim == 2:
        return x, True
    raise ValueError(f"Expected state/actions to be rank-1 or rank-2, got shape {x.shape}")

def _ensure_mask(mask: np.ndarray, dim: int) -> np.ndarray:
    """Make sure flip mask length equals the feature dim."""
    m = np.asarray(mask).reshape(-1)
    if m.size == dim:
        return m
    if m.size < dim:
        pad = np.ones(dim - m.size, dtype=m.dtype)
        return np.concatenate([m, pad], axis=0)
    # m.size > dim
    return m[:dim]

def _apply_gripper_cols(a2d: np.ndarray, fn, idxs=GRIPPER_IDXS) -> np.ndarray:
    """Apply fn on the two gripper columns, preserving shape."""
    cols = np.array(idxs, dtype=int)
    a2d = a2d.copy()
    a2d[:, cols] = fn(a2d[:, cols])
    return a2d

def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """State decode that works for (D,) or (T, D)."""
    s = np.asarray(state)
    if not adapt_to_pi:
        return s

    s2d, batched = _as2d(s)
    D = s2d.shape[1]
    if D < 14:
        raise ValueError(f"_decode_state expects at least 14 dims (got {D}).")

    # 1) 关节翻转
    mask = _ensure_mask(_joint_flip_mask(), D)   # shape: (D,)
    s2d = s2d * mask[None, :]                    # broadcast over T

    # 2) 还原 Aloha runtime 对夹爪的变换
    s2d = _apply_gripper_cols(s2d, _gripper_to_angular)

    return s2d if batched else s2d[0]

def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """Inverse action encode that works for (D,) or (T, D)."""
    a = np.asarray(actions)
    if not adapt_to_pi:
        return a

    a2d, batched = _as2d(a)
    D = a2d.shape[1]
    if D < 14:
        raise ValueError(f"_encode_actions_inv expects at least 14 dims (got {D}).")

    # 1) 关节翻转
    mask = _ensure_mask(_joint_flip_mask(), D)
    a2d = a2d * mask[None, :]

    # 2) 将夹爪从角度域映回（逆变换）
    a2d = _apply_gripper_cols(a2d, _gripper_from_angular_inv)

    return a2d if batched else a2d[0]

@dataclasses.dataclass(frozen=True)
class AlohaInputs_Extra(transforms.DataTransformFn):
    """Inputs for the Aloha policy.
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    action_dim: int = 32

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        # print("BBBBBBB:", data.keys(), data['state'])
        data_ = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        state = transforms.pad_to_dim(state, self.action_dim)
        # print(state.shape)

        # ---- 图像 ----
        in_images = data_["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            # 以前这里是 raise ValueError；多数数据集会包含额外相机键，建议忽略并告警即可
            # 你也可以保留原来的报错逻辑，看你流程需要
            print(f"[AlohaInputs_Extra] Ignoring unexpected cameras: {tuple(unexpected)}")

        if "cam_high" not in in_images:
            # 如果缺基本相机，尝试选一个已有相机做基准；实在没有就给一个兜底黑图
            if in_images:
                base_image = next(iter(in_images.values()))
            else:
                base_image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            base_image = in_images["cam_high"]

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.bool_(True)}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
            "low_0_rgb": "cam_low",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.bool_(True)
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.bool_(False)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data_:
            actions = _encode_actions_inv(np.asarray(data_["actions"]), adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data_:
            inputs["prompt"] = data_["prompt"]
        
        
        # TODO
        if "his_state" in data_ and "data" in data_["his_state"]:
            hs = _decode_state(np.asarray(data_["his_state"]["data"]), adapt_to_pi=self.adapt_to_pi)
            hs = transforms.pad_to_dim(hs, self.action_dim)
            inputs["his_state"] = {**data_["his_state"], "data": hs}
        else:
            inputs["his_state"] = data_.get("his_state", None)

        if "human_action" in data_ and "data" in data_["human_action"]:
            ha = _encode_actions_inv(np.asarray(data_["human_action"]["data"]), adapt_to_pi=self.adapt_to_pi)
            # 若下游把 human_action 当作目标或与 actions 对齐计算，请保留这一行；否则可删掉
            ha = transforms.pad_to_dim(ha, self.action_dim)
            inputs["human_action"] = {**data_["human_action"], "data": ha}
        else:
            inputs["human_action"] = data_.get("human_action", None)


        return inputs
@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    # print(state.shape)
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
