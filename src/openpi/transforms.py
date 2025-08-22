from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

from typing import Any, Literal, Optional, Union
DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        # print(data.keys())
        flat_item = flatten_dict(data)
        # print(flat_item.keys())
        return jax.tree.map(lambda k: flat_item.get(k, None), self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


# @dataclasses.dataclass(frozen=True)
# class Normalize(DataTransformFn):
#     norm_stats: at.PyTree[NormStats] | None
#     # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
#     use_quantiles: bool = False
#     # If true, will raise an error if any of the keys in the norm stats are not present in the data.
#     strict: bool = False

#     def __post_init__(self):
#         if self.norm_stats is not None and self.use_quantiles:
#             _assert_quantile_stats(self.norm_stats)

#     def __call__(self, data: DataDict) -> DataDict:
#         if self.norm_stats is None:
#             return data
        
#         if "his_state" in data and "human_action" in data:
#             print("BBBBBBBBBBB")
#             print(self.norm_stats.keys())
#             self.norm_stats_extra = {
#                 **self.norm_stats,
#                 "his_state":   {"data": self.norm_stats["state"]},   # <- 没有 "data" 这层
#                 "human_action":{"data": self.norm_stats["actions"]},  # <- 同上
#             }

#         applied_dataset = apply_tree(
#             data,
#             self.norm_stats_extra,
#             self._normalize_quantile if self.use_quantiles else self._normalize,
#             strict=self.strict,
#         )
#         print("AAAAAAAAAAA applied:", applied_dataset.keys())
#         return applied_dataset

#     def _normalize(self, x, stats: NormStats):
#         return (x - stats.mean) / (stats.std + 1e-6)

#     def _normalize_quantile(self, x, stats: NormStats):
#         assert stats.q01 is not None
#         assert stats.q99 is not None
#         return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0

import dataclasses

def _align_tail(x, v):
    """把向量 v (D,) reshape 成可与 x 广播的形状 (..., D)。"""
    # 保留 torch.Tensor 类型；否则转 numpy
    try:
        import torch
        if isinstance(x, torch.Tensor):
            v = torch.as_tensor(v, dtype=x.dtype, device=x.device)
        else:
            v = np.asarray(v)
    except Exception:
        v = np.asarray(v)
    if v.ndim == 0:
        return v
    if x.shape[-1] != v.shape[0]:
        raise ValueError(f"尾维不匹配: x.shape={tuple(x.shape)}, v.shape={tuple(v.shape)}")
    return v.reshape((1,) * (x.ndim - 1) + (v.shape[0],))

@dataclasses.dataclass(frozen=True)
class Normalize_Extra(DataTransformFn):
    norm_stats: Mapping[str, Any] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # 调试打印（避免使用 **kwargs）
        # print("Normalize keys:", list(self.norm_stats.keys()))

        # 基于当前 batch 构造局部 stats 视图（不要改 self.norm_stats）
        ns = dict(self.norm_stats)

        # 只有当数据里存在对应结构时，才补齐嵌套 stats 形状
        if isinstance(data.get("his_state"), dict) and "data" in data["his_state"]:
            if "state" in ns:
                ns["his_state"] = {"data": ns["state"]}
        if isinstance(data.get("human_action"), dict) and "data" in data["human_action"]:
            if "actions" in ns:
                ns["human_action"] = {"data": ns["state"]}

        # 严格模式下：只保留顶层在 data 里出现过的键（避免 apply_tree 因多余键报错）
        if self.strict:
            ns = {k: v for k, v in ns.items() if k in data}

        # 应用归一化（应只改有 stats 的叶子，其他键保持原样）
        # TODO
        applied_dataset = apply_tree(
            data,
            ns,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

        # ---- 关键：保留 his_state/human_action 的非 data 元信息 ----
        # 如果 apply_tree 只返回了 {"data": ...} 这样的子树，我们把它和原始字典浅合并
        for k in ("his_state", "human_action"):
            if isinstance(data.get(k), dict) and isinstance(applied_dataset.get(k), dict):
                # 用归一化后的字段覆盖原始同名字段（特别是 "data"），其余元信息保留
                applied_dataset[k] = {**data[k], **applied_dataset[k]}

        # print("Normalize applied keys:", list(applied_dataset.keys()))
        # print("human actions: keys():", applied_dataset["human_action"].keys())
        # print("applied_dataset: human_action:", applied_dataset["human_action"]["data"].shape)
        return applied_dataset

    
    def _normalize(self, x, stats: NormStats):
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0

# @dataclasses.dataclass(frozen=True)
# class NoiseAdd(DataTransformFn):
#     noise_std: float = 1.0
#     mask_ratio: float = 0.1  # 比例，例如 10% 的值会被 mask

#     def __call__(self, data: DataDict) -> DataDict:
#         print("noise add")
#         human_action = data["human_action"]["data"]
#         shape = human_action.shape

#         # 随机生成 mask，True 表示被 mask
#         mask = np.random.rand(*shape) < self.mask_ratio  

#         # 生成噪声
#         noise = np.random.normal(0, self.noise_std, shape)

#         # 应用：mask 位置替换成噪声，其它位置保持原值
#         human_action = np.where(mask, noise, human_action)
#         data["human_action"]["data"] = human_action
#         return data

# TODO 给人的演示轨迹加噪声
@dataclasses.dataclass(frozen=True)
class NoiseAdd:
    noise_std: float = 1.0
    mask_ratio: float = 0.1
    global_std: float = 0.05
    per_seq_std: bool = True
    std_range: tuple[float, float] = (0.5, 1.5)  # 噪声强度缩放范围
    print_once: bool = False     # 是否仅首次打印形状

    def __call__(self, data: DataDict) -> DataDict:
        ha = data.get("human_action", None)

        x = np.asarray(ha["data"])  # (T, D)
        if x.ndim != 2:
            raise ValueError(f"[NoiseAdd] Expected shape (T,D), got {x.shape}")

        T, D = x.shape
        # if self.print_once and not hasattr(self, "_printed"):
        #     print(f"[NoiseAdd] human_action shape: (T={T}, D={D})")
        #     object.__setattr__(self, "_printed", True)

        # ---- 局部替换噪声 ----
        local_noise = np.random.normal(0.0, self.noise_std, size=x.shape).astype(x.dtype, copy=False)
        replace_mask = np.random.rand(*x.shape) < self.mask_ratio
        x_noisy = np.where(replace_mask, local_noise, x)

        # ---- 全局加性噪声 ----
        if self.per_seq_std:
            scale = np.random.uniform(self.std_range[0], self.std_range[1])
        else:
            scale = 1.0
        global_noise = np.random.normal(0.0, self.global_std, size=x.shape).astype(x.dtype) * scale
        x_noisy = x_noisy + global_noise

        # ---- 写回 ----
        data = dict(data)
        data["human_action"] = dict(ha)
        data["human_action"]["data"] = x_noisy
        return data



@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        return x * (stats.std + 1e-6) + stats.mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data

@dataclasses.dataclass(frozen=True)
class ExtraActions(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        data["human_action"] = data["human_action"]
        data["his_state"] = data["his_state"]
        # print("extra: ", data["human_action"]["data"].shape)
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
