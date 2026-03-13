# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "HsdConfig", "KLControlConfig", "OppoConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class OppoConfig(BaseConfig):
    """Configuration for OPPO-style Bayesian advantage.

    Args:
        rbar_eps (float): Clamp value for prior success rate to avoid inf logit.
        cumsum_decay (float): Decay factor for discounted cumulative sum of token-level
            evidence. 1.0 = plain cumsum; 0.999 gives an effective window of ~1000 tokens,
            bounding the logit range and preventing sigmoid saturation on long sequences.
        whiten_adv (bool): Apply masked whitening (mean=0, std=1) to advantages.
            Stabilises gradient magnitudes but discards the original probability-change
            scale inherent in OPPO advantages. Set False to preserve absolute semantics.
        oracle_prompt_prefix (str): Prefix added before ground truth in oracle prompt.
        oracle_prompt_suffix (str): Suffix added after ground truth in oracle prompt.
        oracle_prompt_sep (str): Separator inserted before oracle prefix/ground truth.
        truncation (str): Truncation policy when oracle prompt exceeds max prompt length.
    """

    rbar_eps: float = 1e-4
    cumsum_decay: float = 1.0
    whiten_adv: bool = False
    oracle_prompt_prefix: str = ""
    oracle_prompt_suffix: str = ""
    oracle_prompt_sep: str = " "
    truncation: str = "left"


@dataclass
class HsdConfig(BaseConfig):
    """Configuration for Hindsight Self-Distillation (HSD).

    HSD uses a teacher signal (oracle-conditioned on ground truth y*) to distil
    knowledge into the student via KL divergence on full logits.  An auxiliary PPO
    loss with GRPO-style reward-based advantages provides the reward-driven signal.

    Args:
        kl_coef (float): Weight of the KL distillation loss term.
        ppo_coef (float): Weight of the auxiliary PPO clipped loss. 0 = disabled.
        reward_gate_kl (bool): Only apply KL distillation on correct rollouts
            (sequence reward > 0). Prevents distilling from incorrect reasoning.
        oracle_prompt_prefix (str): Prefix added before ground truth in oracle prompt.
        oracle_prompt_suffix (str): Suffix added after ground truth in oracle prompt.
        oracle_prompt_sep (str): Separator before oracle prefix/ground truth.
        truncation (str): Truncation policy when oracle prompt exceeds max prompt length.
        temperature (float): Temperature for softening teacher logits in KL loss.
    """

    kl_coef: float = 0.3
    ppo_coef: float = 1.0
    reward_gate_kl: bool = True
    oracle_prompt_prefix: str = ""
    oracle_prompt_suffix: str = ""
    oracle_prompt_sep: str = " "
    truncation: str = "left"
    temperature: float = 1.0


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        rollout_is_threshold (Optional[float]): Upper threshold for IS weights. null = disabled,
            float value = enabled (compute weights and metrics). This is the main on/off switch.
        rollout_is_threshold_lower (Optional[float]): Lower threshold for IS weights. If None, defaults to 1/upper.
        rollout_is_level (str): Aggregation level: "token", "sequence", or "geometric".
        rollout_is_mode (str): Bounding mode: "truncate" (cap upper only) or "mask" (zero outside bounds).
        rollout_is_veto_threshold (float): Per-token veto threshold for catastrophic outliers.
        rollout_is (bool): Whether to apply IS weights to policy loss. True = apply weights,
            False = compute metrics only (useful for monitoring before enabling correction). Default: False.
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
    # Rollout Importance Sampling (replaces legacy tis_imp_ratio_cap)
    # Controls computation of IS weights and mismatch metrics
    rollout_is_threshold: Optional[float] = None  # null = disabled, float = enabled
    rollout_is_threshold_lower: Optional[float] = None
    rollout_is_level: str = "token"
    rollout_is_mode: str = "truncate"
    rollout_is_veto_threshold: Optional[float] = 1e-4
    # Controls whether to apply IS weights to policy loss (only if rollout_is_threshold is set)
    # True = apply weights to loss, False = compute metrics only (no weight application)
    rollout_is: bool = False
    oppo: OppoConfig = field(default_factory=OppoConfig)
    hsd: HsdConfig = field(default_factory=HsdConfig)
