import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""
        
    @abc.abstractmethod
    def infer_realtime(self, obs: Dict, prev_action_chunk: jax.Array,
                      inference_delay: int,
                      prefix_attention_horizon: int,
                      prefix_attention_schedule,
                      max_guidance_weight: float) -> Dict:

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
