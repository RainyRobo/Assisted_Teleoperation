from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._sample_rtc_actions = nnx_utils.module_jit(model.sample_actions_rtc)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self.prev_action_chunk = None

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        inputs = self._input_transform(inputs)

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        shapes = jax.tree.map(lambda x: x.shape, inputs)
        # print("Input shapes3:", shapes)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }
        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs
    
    @override
    def infer_realtime(self, obs: dict, prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule,
        max_guidance_weight: float):
        """Infer actions in real-time with a prefix attention schedule."""
                # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        shapes = jax.tree.map(lambda x: x.shape, inputs)
        if self.prev_action_chunk is not None:
            prev_action_chunk = np.concatenate([ self.prev_action_chunk[inference_delay:], np.zeros((inference_delay, 14)),], axis=0)
        else:
            prev_action_chunk = np.zeros((50, 14))
       
        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self._sample_rtc_actions(sample_rng, _model.Observation.from_dict(inputs), inference_delay=inference_delay,
                prefix_actions=prev_action_chunk, prefix_attention_horizon=prefix_attention_horizon,
                prefix_attention_schedule=prefix_attention_schedule, max_guidance_weight=max_guidance_weight, **self._sample_kwargs)
        # print("AAAAA: actions", actions[0, :, 0], "shape", actions.shape)
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        self.prev_action_chunk = np.asarray(jax.device_get(
            outputs["actions"][:, :14]
        ))
        model_time = time.monotonic() - start_time
        outputs = self._output_transform(outputs)

        # print("AAAAA: outputs", outputs["actions"][:, 0], "shape", outputs["actions"].shape)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs
        

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
