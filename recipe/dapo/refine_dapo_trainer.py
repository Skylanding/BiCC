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
"""
RefineDAPOTrainer - DAPO trainer
Inherits from RayDAPOTrainer
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

from .dapo_ray_trainer import RayDAPOTrainer


# ============================================================================
# RefineDAPOTrainer
# ============================================================================


class RefineDAPOTrainer(RayDAPOTrainer):
    """
    DAPO trainer with optional Contrastive GRPO support
    
    Inherits from RayDAPOTrainer and provides standard DAPO training functionality.
    Optionally supports Contrastive GRPO, which reformulates GRPO as a contrastive 
    learning objective (similar to DPO format):
    
    J_GRPO ∝ (1/n^+) * Σ_{o_j ∈ O^+} log π_θ(o_j|q) - (1/n^-) * Σ_{o_k ∈ O^-} log π_θ(o_k|q)
    
    In the special case (n=2 with one positive, one negative):
    J_2-GRPO = log π_θ(o^+|q) - log π_θ(o^-|q) = log[π_θ(o^+|q) / π_θ(o^-|q)]
    
    This is identical to DPO's core term!
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load contrastive GRPO config from top-level config (not trainer)
        cgrpo_config = getattr(self.config, 'contrastive_grpo', None)
        
        if cgrpo_config is not None:
            self.cgrpo_config = ContrastiveGRPOConfig(**cgrpo_config)
            self.use_contrastive_grpo = self.cgrpo_config.enable
        else:
            # Default configuration
            self.cgrpo_config = ContrastiveGRPOConfig(enable=False)
            self.use_contrastive_grpo = False
        
        # Load GRPO-SRC config from top-level config (not trainer)
        cgrpo_src_config = getattr(self.config, 'contrastive_grpo_src', None)
        
        if cgrpo_src_config is not None:
            self.cgrpo_src_config = ContrastiveGRPOSRCConfig(**cgrpo_src_config)
            self.use_contrastive_grpo_src = self.cgrpo_src_config.enable
        else:
            # Default configuration
            self.cgrpo_src_config = ContrastiveGRPOSRCConfig(enable=False)
            self.use_contrastive_grpo_src = False
        
        # SRC requires contrastive GRPO to be enabled
        if self.use_contrastive_grpo_src and not self.use_contrastive_grpo:
            print("⚠️  Warning: GRPO-SRC requires Contrastive GRPO to be enabled.")
            print("   Disabling GRPO-SRC. Please enable contrastive_grpo.enable=True first.")
            self.use_contrastive_grpo_src = False
    
    def _truncate_batch_for_data_parallelism(self, batch: DataProto, batch_name: str = "batch") -> DataProto:
        """
        Truncate batch to be divisible by n_gpus for data parallelism.
        
        Args:
            batch: Batch to truncate
            batch_name: Name of batch for warning messages
            
        Returns:
            Truncated batch
        """
        n_gpus = self.config.trainer.n_gpus_per_node
        batch_size = len(batch)
        if batch_size % n_gpus != 0:
            truncated_size = (batch_size // n_gpus) * n_gpus
            if truncated_size > 0:
                batch = batch[:truncated_size]
                print(f"Warning: Truncated {batch_name} from {batch_size} to {truncated_size} to be divisible by {n_gpus} GPUs")
            else:
                print(f"Warning: {batch_name} size {batch_size} is too small for {n_gpus} GPUs")
        return batch
    
    def _check_conditioned_batch_size(self, conditioned_batch: DataProto, n_gpus: int) -> bool:
        """
        Check if conditioned batch has enough samples for data parallelism.
        
        Args:
            conditioned_batch: Conditioned batch to check
            n_gpus: Number of GPUs
            
        Returns:
            True if batch is large enough, False otherwise
        """
        if conditioned_batch is None:
            return False
        conditioned_batch_size = len(conditioned_batch.batch['input_ids']) if 'input_ids' in conditioned_batch.batch else 0
        return conditioned_batch_size >= n_gpus

        if self.use_contrastive_grpo:
            print("\n" + "="*80)
            if self.use_contrastive_grpo_src:
                print("🎯 Contrastive GRPO with Self-Reflective Conditioning (GRPO-SRC) Enabled")
            else:
                print("🎯 Contrastive GRPO Enabled in RefineDAPOTrainer")
            print("="*80)
            print("Mathematical Framework:")
            print("  • Partitioning: O^+ = {o: r=1}, O^- = {o: r=0}")
            print("  • Advantages: A^+ = sqrt(n^-/n^+), A^- = -sqrt(n^+/n^-)")
            if self.use_contrastive_grpo_src:
                print("  • Self-Reflective Conditioning: π(y | x, y_alt)")
                print("    - Positive: log π(yw | x, yl)  (看到错误，生成正确)")
                print("    - Negative: log π(yl | x, yw)  (看到正确，识别错误)")
            else:
                print("  • Objective: J ∝ (1/n^+)Σlog π(o^+) - (1/n^-)Σlog π(o^-)")
                print(f"  • Special case (n=2): J = log[π(o^+)/π(o^-)]  [Same as DPO!]")
            print("="*80 + "\n")

    def fit(self):
        """
        Training loop
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Print training summary
        print("\n" + "="*80)
        print("🚀 Starting Training")
        print("="*80)
        print(f"  Total training steps: {self.total_training_steps}")
        print(f"  Total epochs: {self.config.trainer.total_epochs}")
        print(f"  Train batch size: {self.config.data.train_batch_size}")
        print(f"  Rollout samples per prompt: {self.config.actor_rollout_ref.rollout.n}")
        print(f"  Advantage estimator: {self.config.algorithm.adv_estimator}")
        if self.use_contrastive_grpo:
            print(f"  ✅ Contrastive GRPO: ENABLED (DPO-like format)")
        else:
            print(f"  ❌ Contrastive GRPO: DISABLED")
        print(f"  Validation before train: {self.config.trainer.get('val_before_train', False)}")
        print(f"  Validation frequency: {self.config.trainer.get('test_freq', 0)}")
        print(f"  Save frequency: {self.config.trainer.get('save_freq', 0)}")
        print("="*80 + "\n")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        # Debug: Check reward values
                        if reward_tensor is not None:
                            seq_rewards_sum = reward_tensor.sum(dim=-1) if reward_tensor.dim() > 1 else reward_tensor
                            reward_mean = seq_rewards_sum.mean().item()
                            reward_std = seq_rewards_sum.std().item()
                            reward_min = seq_rewards_sum.min().item()
                            reward_max = seq_rewards_sum.max().item()
                            
                            # Print warning if rewards are suspicious
                            if reward_mean == 0.0 and reward_std == 0.0:
                                print(f"⚠️  WARNING: All rewards are 0! reward_tensor shape: {reward_tensor.shape}, "
                                      f"mean={reward_mean}, std={reward_std}")
                            elif abs(reward_mean) < 1e-6 and reward_std < 1e-6:
                                print(f"⚠️  WARNING: Rewards are extremely small! mean={reward_mean:.6f}, std={reward_std:.6f}")
                            elif self.global_steps % 10 == 0:  # Print every 10 steps for debugging
                                print(f"🔍 Step {self.global_steps} Reward stats: "
                                      f"mean={reward_mean:.4f}, std={reward_std:.4f}, "
                                      f"min={reward_min:.4f}, max={reward_max:.4f}")

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # Apply DAPO's filter groups logic
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        # DAPO's dynamic sampling and group filtering
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "acc":
                            # For 'acc' metric, try to extract from reward_extra_info or use seq_reward as fallback
                            if "acc" in new_batch.non_tensor_batch:
                                # acc already exists in batch
                                pass
                            else:
                                # Fallback: use seq_reward if acc is not available
                                print(f"⚠️  Warning: metric='acc' not found in batch, falling back to 'seq_reward'")
                                new_batch.non_tensor_batch["acc"] = (
                                    new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                                )
                                metric_name = "acc"  # Update to use the created metric

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        # For binary metrics like 'acc' (0/1), keep all prompts
                        # because even prompts with uniform acc values (all 0 or all 1) 
                        # have learning value for RLVR tasks
                        if metric_name == "acc":
                            # For binary acc, keep all prompts regardless of std
                            kept_prompt_uids = list(prompt_uid2metric_std.keys())
                        else:
                            # For continuous metrics, use std-based filtering
                            kept_prompt_uids = [
                                uid
                                for uid, std in prompt_uid2metric_std.items()
                                if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                            ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            total_prompts = len(prompt_uid2metric_std)
                            kept_prompts = len(kept_prompt_uids)
                            filtered_prompts = total_prompts - kept_prompts
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            print(f"  Total prompts generated: {total_prompts}, Kept: {kept_prompts}, Filtered: {filtered_prompts}")
                            if total_prompts > 0:
                                avg_std = np.mean(list(prompt_uid2metric_std.values()))
                                print(f"  Average metric std: {avg_std:.6f}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # === GRPO-SRC: Early advantage computation for SRC conditioning ===
                    # For GRPO-SRC, we need to compute advantages first to determine positive/negative
                    # samples, then create conditioned inputs before computing log_probs.
                    #
                    # Theoretical Note:
                    # We use advantages computed from rewards under π(y|x) to weight log probabilities
                    # under π(y|x,y_alt). This assumes reward r(x,y) is outcome-based and doesn't
                    # depend on the generation process/conditioning context.
                    src_conditioned_batch = None
                    baseline_batch_for_comparison = None  # Save original batch for baseline comparison
                    if self.use_contrastive_grpo_src:
                        device = batch.batch["token_level_rewards"].device
                        rollout_n = self.config.actor_rollout_ref.rollout.n
                        # 将rollout_n传递给config，以便在create_self_reflective_pairs中使用
                        self.cgrpo_src_config.rollout_n = rollout_n
                        # Compute advantages and create SRC pairs
                        batch, cgrpo_src_metrics = format_grpo_as_contrastive_with_src(
                            batch=batch,
                            config=self.cgrpo_src_config,
                            device=device
                        )
                        # Store metrics
                        if 'contrastive_grpo_src' not in batch.meta_info:
                            batch.meta_info['contrastive_grpo_src'] = cgrpo_src_metrics
                        
                        # If SRC pairs were created, prepare conditioned batch for log_prob computation
                        if 'src_pairing_info' in batch.meta_info:
                            pairing_info = batch.meta_info['src_pairing_info']
                            # pairing_info['conditioned_input_ids'] is always a Tensor (stacked in create_self_reflective_pairs)
                            if isinstance(pairing_info['conditioned_input_ids'], torch.Tensor) and len(pairing_info['conditioned_input_ids']) > 0:
                                conditioned_input_ids = pairing_info['conditioned_input_ids']
                                conditioned_attention_mask = pairing_info['conditioned_attention_mask']
                                conditioned_response_mask = pairing_info['conditioned_response_mask']
                                # original_indices is always a tensor (converted in create_self_reflective_pairs)
                                if isinstance(pairing_info['original_indices'], torch.Tensor):
                                    original_indices_list = pairing_info['original_indices'].cpu().tolist()
                                else:
                                    original_indices_list = list(pairing_info['original_indices'])
                                
                                # Extract responses from conditioned_input_ids using response_mask
                                batch_size, seq_len = conditioned_input_ids.shape
                                conditioned_responses_list = []
                                for i in range(batch_size):
                                    resp_mask = conditioned_response_mask[i].bool()
                                    if resp_mask.any():
                                        response_tokens = conditioned_input_ids[i][resp_mask]
                                        conditioned_responses_list.append(response_tokens)
                                    else:
                                        conditioned_responses_list.append(torch.empty(0, device=device, dtype=conditioned_input_ids.dtype))
                                
                                # Find max response length and pad all responses to same length
                                max_response_len = max(len(r) for r in conditioned_responses_list) if conditioned_responses_list else 0
                                if max_response_len > 0:
                                    pad_token_id = 0
                                    conditioned_responses = torch.stack([
                                        torch.cat([r, torch.full((max_response_len - len(r),), pad_token_id, device=device, dtype=r.dtype)])
                                        if len(r) < max_response_len else r
                                        for r in conditioned_responses_list
                                    ])
                                else:
                                    conditioned_responses = torch.empty((batch_size, 0), device=device, dtype=conditioned_input_ids.dtype)
                                
                                # Create position_ids
                                conditioned_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
                                
                                # Copy necessary meta_info from original batch
                                conditioned_meta_info = deepcopy(batch.meta_info)
                                conditioned_meta_info['max_token_len'] = max(conditioned_meta_info.get('max_token_len', seq_len), seq_len)
                                
                                # Create a new batch with conditioned inputs
                                src_conditioned_batch = DataProto.from_dict(
                                    tensors={
                                        'input_ids': conditioned_input_ids,
                                        'attention_mask': conditioned_attention_mask,
                                        'response_mask': conditioned_response_mask,
                                        'responses': conditioned_responses,
                                        'position_ids': conditioned_position_ids,
                                    },
                                    non_tensors={
                                        'original_indices': np.array(original_indices_list),
                                    },
                                    meta_info=conditioned_meta_info
                                )
                                
                                # Truncate to be divisible by n_gpus (for data parallelism)
                                n_gpus = self.config.trainer.n_gpus_per_node
                                batch_size = len(src_conditioned_batch)
                                if batch_size % n_gpus != 0:
                                    truncated_size = (batch_size // n_gpus) * n_gpus
                                    if truncated_size > 0:
                                        src_conditioned_batch = src_conditioned_batch[:truncated_size]
                                        original_indices_list = original_indices_list[:truncated_size]
                                        src_conditioned_batch.non_tensor_batch['original_indices'] = np.array(original_indices_list)
                                        print(f"Warning: Truncated src_conditioned_batch from {batch_size} to {truncated_size} to be divisible by {n_gpus} GPUs")
                                    else:
                                        print(f"Warning: src_conditioned_batch size {batch_size} is too small for {n_gpus} GPUs. Setting to None.")
                                        src_conditioned_batch = None
                                
                                # Create mapping from original to conditioned indices
                                if src_conditioned_batch is not None:
                                    batch.meta_info['src_original_to_conditioned'] = torch.tensor(original_indices_list, device=device, dtype=torch.long)
                            else:
                                # No pairs created
                                src_conditioned_batch = None

                    # recompute old_log_probs
                    # If using SRC, compute log_prob on conditioned inputs, then map back to original indices
                    # Save baseline batch AFTER computing log_probs (for accurate baseline comparison)
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        if self.use_contrastive_grpo_src and src_conditioned_batch is not None:
                            n_gpus = self.config.trainer.n_gpus_per_node
                            
                            # If conditioned batch is too small for data parallelism, use original batch
                            if not self._check_conditioned_batch_size(src_conditioned_batch, n_gpus):
                                conditioned_batch_size = len(src_conditioned_batch.batch['input_ids']) if src_conditioned_batch and 'input_ids' in src_conditioned_batch.batch else 0
                                print(f"Warning: src_conditioned_batch has only {conditioned_batch_size} samples, less than {n_gpus} GPUs. Using original batch for log_prob computation.")
                                batch = self._truncate_batch_for_data_parallelism(batch, "original batch")
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            else:
                                # First, compute log_prob on original batch (for unpaired samples)
                                batch = self._truncate_batch_for_data_parallelism(batch, "original batch")
                                original_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                original_log_probs_all = original_log_prob.batch.get("log_probs")
                                original_entropys_all = original_log_prob.batch.get("entropys")
                                
                                # Save baseline batch with original log_probs (for baseline comparison)
                                # This is the batch state before SRC conditioning, with original log_probs
                                if baseline_batch_for_comparison is None:
                                    baseline_batch_for_comparison = deepcopy(batch)
                                    # Add original log_probs to baseline batch
                                    baseline_batch_for_comparison = baseline_batch_for_comparison.union(original_log_prob)
                                
                                # Compute log_prob on conditioned inputs
                                # For GRPO-SRC: We keep the original batch structure (GRPO's pair representation),
                                # but compute log_prob using conditioned inputs, then map back to original batch indices.
                                # This is just rewriting the expression with conditioning, not changing the batch structure.
                                conditioned_log_prob = self.actor_rollout_wg.compute_log_prob(src_conditioned_batch)
                                conditioned_log_probs = conditioned_log_prob.batch.get("log_probs", None)
                                
                                if conditioned_log_probs is not None:
                                    # Map conditioned log_probs back to original batch indices
                                    # Each conditioned sample corresponds to one original sample in the GRPO batch
                                    original_indices = batch.meta_info['src_original_to_conditioned'].cpu().numpy()
                                    
                                    # Verify data consistency
                                    if len(original_indices) != len(conditioned_log_probs):
                                        raise ValueError(
                                            f"Mismatch between original_indices ({len(original_indices)}) "
                                            f"and conditioned_log_probs ({len(conditioned_log_probs)})"
                                        )
                                    
                                    # Start with original log_probs (for unpaired samples)
                                    final_log_probs = original_log_probs_all.clone()
                                    
                                    # Map conditioned log_probs to original batch indices
                                    # Each conditioned sample maps to one original sample
                                    for cond_idx, orig_idx in enumerate(original_indices):
                                        if 0 <= orig_idx < len(final_log_probs) and cond_idx < len(conditioned_log_probs):
                                            final_log_probs[orig_idx] = conditioned_log_probs[cond_idx]
                                    
                                    # Handle entropys similarly
                                    if original_entropys_all is not None:
                                        final_entropys = original_entropys_all.clone()
                                        if "entropys" in conditioned_log_prob.batch:
                                            conditioned_entropys = conditioned_log_prob.batch["entropys"]
                                            for cond_idx, orig_idx in enumerate(original_indices):
                                                if 0 <= orig_idx < len(final_entropys) and cond_idx < len(conditioned_entropys):
                                                    final_entropys[orig_idx] = conditioned_entropys[cond_idx]
                                        else:
                                            final_entropys = torch.zeros_like(final_log_probs)
                                    else:
                                        final_entropys = torch.zeros_like(final_log_probs)
                                    
                                    old_log_prob = DataProto.from_dict(
                                        tensors={
                                            'log_probs': final_log_probs,
                                            'entropys': final_entropys
                                        }
                                    )
                                else:
                                    # Fallback: use original log_prob if computation fails
                                    old_log_prob = original_log_prob
                        else:
                            # Standard log_prob computation
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        

                    if self.use_reference_policy:
                        # compute reference log_prob
                        # For GRPO-SRC: According to the formula, ref_log_prob should use ORIGINAL inputs (q),
                        # not conditioned inputs ([q;o^-] or [q;o^+]).
                        # Formula: ρ^{+(j)} = π_θ(o^+|[q;o^-]) / π_{θ_old}(o^+|q)
                        #         ρ^{-(i)} = π_θ(o^-|[q;o^+]) / π_{θ_old}(o^-|q)
                        # The denominator π_{θ_old}(o|q) must be computed on original inputs, not conditioned inputs.
                        with marked_timer("ref", timing_raw, "olive"):
                            if self.use_contrastive_grpo_src and src_conditioned_batch is not None:
                                n_gpus = self.config.trainer.n_gpus_per_node
                                
                                # If conditioned batch is too small for data parallelism, use original batch
                                if not self._check_conditioned_batch_size(src_conditioned_batch, n_gpus):
                                    conditioned_batch_size = len(src_conditioned_batch.batch['input_ids']) if src_conditioned_batch and 'input_ids' in src_conditioned_batch.batch else 0
                                    print(f"Warning: src_conditioned_batch has only {conditioned_batch_size} samples, less than {n_gpus} GPUs. Using original batch for ref_log_prob computation.")
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    # For GRPO-SRC, ref_log_prob should use ORIGINAL inputs (q), not conditioned inputs ([q;o^-] or [q;o^+]).
                                    # This is because the formula requires: ρ = π_θ(o|[q;o_alt]) / π_{θ_old}(o|q)
                                    # The denominator π_{θ_old}(o|q) must be computed on original inputs.
                                    # 
                                    # Since we keep the original batch structure, we can compute ref_log_prob directly on the original batch.
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                # Standard reference log_prob computation
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout IS weights and mismatch metrics
                    batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                    metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        # Use contrastive formulation if enabled (but not SRC, as SRC already computed above)
                        if self.use_contrastive_grpo and not self.use_contrastive_grpo_src:
                            device = batch.batch["token_level_rewards"].device
                            batch, cgrpo_metrics = format_grpo_as_contrastive(
                                batch=batch,
                                config=self.cgrpo_config,
                                device=device
                            )
                            # Store metrics for logging
                            if 'contrastive_grpo' not in batch.meta_info:
                                batch.meta_info['contrastive_grpo'] = cgrpo_metrics
                        elif not self.use_contrastive_grpo:
                            # Standard advantage computation
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )
                        # Note: If use_contrastive_grpo_src, advantages were already computed above

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)
                            
                            # === Compute baseline vs SRC-GRPO gain ===
                            # For SRC-GRPO, compute baseline (standard GRPO) loss for comparison
                            if self.use_contrastive_grpo_src and baseline_batch_for_comparison is not None:
                                # SRC-GRPO loss is already computed above (actor_output_metrics['actor/loss'])
                                src_loss = actor_output_metrics.get('actor/loss', 0.0)
                                
                                # Store SRC loss
                                metrics['grpo_src/src_loss'] = src_loss
                                
                                # Compute baseline (standard GRPO) loss using original batch (before SRC modifications)
                                # We use the saved baseline_batch_for_comparison which has the original state
                                baseline_batch = deepcopy(baseline_batch_for_comparison)
                                
                                # Compute standard GRPO advantages (not contrastive, not SRC)
                                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                                baseline_batch = compute_advantage(
                                    baseline_batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                )
                                
                                # Estimate baseline loss using standard GRPO advantages
                                # This is a simplified estimate - for exact comparison, we'd need to recompute
                                # the full actor update, but that's too expensive
                                baseline_advantages = baseline_batch.batch.get("advantages", None)
                                if baseline_advantages is not None:
                                    # Estimate baseline loss: -mean(advantages * log_probs)
                                    # This is a simplified version, actual loss includes clipping, etc.
                                    # Use log_probs from baseline batch (these are the original log_probs, not conditioned)
                                    baseline_log_probs = baseline_batch.batch.get("log_probs", None)
                                    if baseline_log_probs is None:
                                        # Fallback: use from current batch if not available (shouldn't happen)
                                        baseline_log_probs = batch.batch.get("log_probs", None)
                                        print("⚠️  Warning: baseline_log_probs not found in baseline_batch, using current batch (approximation)")
                                    
                                    if baseline_log_probs is not None:
                                        response_mask_baseline = baseline_batch.batch.get("response_mask", None)
                                        if response_mask_baseline is not None:
                                            # Compute token-level loss estimate
                                            masked_advantages = baseline_advantages * response_mask_baseline.float()
                                            masked_log_probs = baseline_log_probs * response_mask_baseline.float()
                                            # Simplified baseline loss estimate (negative of advantage-weighted log prob)
                                            baseline_loss_estimate = -(masked_advantages * masked_log_probs).sum() / (response_mask_baseline.sum() + 1e-8)
                                            baseline_loss = baseline_loss_estimate.item()
                                            
                                            # Compute gain: positive gain means SRC-GRPO is better (lower loss is better)
                                            # gain = baseline_loss - src_loss
                                            # If gain > 0: SRC-GRPO has lower loss (better)
                                            # If gain < 0: Baseline has lower loss (better)
                                            gain = baseline_loss - src_loss
                                            
                                            metrics['grpo_src/baseline_loss'] = baseline_loss
                                            metrics['grpo_src/gain'] = gain
                                            metrics['grpo_src/gain_percent'] = (gain / (abs(baseline_loss) + 1e-8)) * 100
                                            
                                            # Log gain information
                                            if self.global_steps % 10 == 0:
                                                print(f"📊 SRC-GRPO Gain Analysis (Step {self.global_steps}):")
                                                print(f"   Baseline (Standard GRPO) Loss: {baseline_loss:.6f}")
                                                print(f"   SRC-GRPO Loss: {src_loss:.6f}")
                                                print(f"   Gain: {gain:.6f} ({metrics['grpo_src/gain_percent']:.2f}%)")
                                                if gain > 0:
                                                    print(f"   ✅ SRC-GRPO improves over baseline by {gain:.6f} (lower loss is better)")
                                                else:
                                                    print(f"   ⚠️  Baseline is better by {abs(gain):.6f}")

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    print(f"\n🔍 Running validation at step {self.global_steps}...")
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    # Print key validation metrics
                    val_reward = val_metrics.get('val/mean_reward', val_metrics.get('val/mean_seq_reward', 'N/A'))
                    print(f"✅ Validation completed: mean_reward={val_reward}")

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    print(f"💾 Saving checkpoint at step {self.global_steps}...")
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()
                    print(f"✅ Checkpoint saved successfully")

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                # Add contrastive GRPO metrics if enabled (must be before setting batch=None)
                if self.use_contrastive_grpo and batch is not None and 'contrastive_grpo' in batch.meta_info:
                    cgrpo_metrics = batch.meta_info['contrastive_grpo']
                    for key, value in cgrpo_metrics.items():
                        metrics[f'contrastive_grpo/{key}'] = value
                
                # Add GRPO-SRC metrics if enabled (must be before setting batch=None)
                if self.use_contrastive_grpo_src and batch is not None and 'contrastive_grpo_src' in batch.meta_info:
                    cgrpo_src_metrics = batch.meta_info['contrastive_grpo_src']
                    for key, value in cgrpo_src_metrics.items():
                        metrics[f'grpo_src/{key}'] = value
                
                metrics["contrastive_grpo/enabled"] = 1.0 if self.use_contrastive_grpo else 0.0
                metrics["grpo_src/enabled"] = 1.0 if self.use_contrastive_grpo_src else 0.0
                metrics["train/num_gen_batches"] = num_gen_batches
                
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)
                
                # Print step summary every 5 steps
                if self.global_steps % 5 == 0:
                    reward_mean = metrics.get('data/mean_seq_reward', 0.0)
                    reward_std = metrics.get('data/std_seq_reward', 0.0)
                    actor_loss = metrics.get('actor/loss', 0.0)
                    print(f"📍 Step {self.global_steps}: "
                          f"reward={reward_mean:.4f}±{reward_std:.4f}, "
                          f"actor_loss={actor_loss:.6f}")
                    
                    if self.use_contrastive_grpo:
                        n_groups = metrics.get('contrastive_grpo/total_groups', 0)
                        avg_ratio = metrics.get('contrastive_grpo/avg_pos_neg_ratio', 0.0)
                        print(f"   Contrastive GRPO: {n_groups} groups, avg_pos/neg={avg_ratio:.3f}")

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        
        # check if last step checkpoint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)


# ============================================================================
# Contrastive GRPO Trainer - Reformulates GRPO as a contrastive learning objective
# ============================================================================

"""
Contrastive GRPO Trainer - Reformulates GRPO as a contrastive learning objective

Mathematical Framework:

    Given n rollouts {o_1, ..., o_n} with binary rewards {r_1, ..., r_n} ∈ {0,1}^n:

    

    1. Partition into positive and negative sets:

        O^+ = {o_i : r_i = 1}  (correct responses)

        O^- = {o_i : r_i = 0}  (incorrect responses)

    

    2. Compute advantages (derived from GRPO's group normalization):

        A^+ = sqrt(n^- / n^+)

        A^- = -sqrt(n^+ / n^-)

    

    3. GRPO objective in contrastive form:

        J_GRPO ∝ (1/n^+) * Σ_{o_j ∈ O^+} log π_θ(o_j|q) - (1/n^-) * Σ_{o_k ∈ O^-} log π_θ(o_k|q)

    

    4. Special case (n=2 with one positive, one negative):

        J_2-GRPO = log π_θ(o^+|q) - log π_θ(o^-|q) = log[π_θ(o^+|q) / π_θ(o^-|q)]

        

        This is identical to DPO's core term!



Key Insight:

    GRPO is already a contrastive learning algorithm. We don't need to add DPO loss;

    we just need to properly compute and apply the contrastive advantages derived from

    GRPO's group normalization.
"""


# ============================================================================
# Contrastive GRPO with Self-Reflective Conditioning (GRPO-SRC)
# ============================================================================
# NOTE: The following GRPO-SRC implementation is provided but not integrated.
# It can be enabled in the future if needed. Currently, only Contrastive GRPO
# is integrated into RefineDAPOTrainer.
#
# GRPO-SRC combines:
# 1. Contrastive GRPO: A+ = √(n-/n+), A- = -√(n+/n-)
# 2. SRC: Conditioning on alternative responses
#
# Formula: J_GRPO-SRC = Σ_i A_i * log π_θ(o_i | x, o_alt)


@dataclass
class ContrastiveGRPOSRCConfig:
    """
    Contrastive GRPO with Self-Reflective Conditioning配置
    
    结合两种增强：
    1. Contrastive advantages (从GRPO数学推导)
    2. Self-reflective conditioning (从SRC)
    """
    
    # 基础配置
    enable: bool = True
    epsilon: float = 1e-8
    
    # SRC配置
    use_self_reflection: bool = True  # 是否启用self-reflection
    symmetric_conditioning: bool = True  # 对称交叉条件
    
    # 退化情况处理
    handle_uniform_rewards: bool = True
    uniform_reward_advantage: float = 0.0
    
    # 日志
    log_advantage_stats: bool = True
    log_src_metrics: bool = True
    verify_equivalence: bool = False
    
    # Rollout配置（用于优化配对策略）
    rollout_n: int = 2  # Rollout数量，当为2时只创建一对配对（符合数学假设）


@dataclass
class ContrastiveGRPOConfig:
    """
    Configuration for Contrastive GRPO.
    
    This formulation reveals GRPO's inherent contrastive structure without
    adding any external loss terms. The advantages are computed exactly as
    GRPO does, but we make the positive/negative partitioning explicit.
    """
    
    # Enable contrastive formulation (if False, falls back to standard GRPO)
    enable: bool = True
    
    # Numerical stability
    epsilon: float = 1e-8
    
    # Degenerate case handling (when all rewards are identical)
    handle_uniform_rewards: bool = True
    uniform_reward_advantage: float = 0.0  # Advantage when all rewards are same
    
    # Logging and debugging
    log_advantage_stats: bool = True
    log_contrastive_structure: bool = True
    
    # Verification mode: check that contrastive formulation matches standard GRPO
    verify_equivalence: bool = False


def compute_contrastive_advantages(
    rewards: torch.Tensor,
    group_indices: List[int],
    config: ContrastiveGRPOConfig,
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute advantages using GRPO's contrastive structure.
    
    Mathematical Derivation:
    ------------------------
    Given rewards r_1, ..., r_n for a group:
    
    Standard GRPO computes:

        A_i = (r_i - r_bar) / (std(r) + ε)
    
    For binary rewards:

        r_bar = n^+ / n

        std(r) = sqrt((n^+ * n^-) / n^2) = sqrt(n^+ * n^-) / n
    
    Therefore:

        A^+ = (1 - n^+/n) / (sqrt(n^+ * n^-)/n) = n^- / sqrt(n^+ * n^-)

            = sqrt(n^- / n^+)
        
        A^- = (0 - n^+/n) / (sqrt(n^+ * n^-)/n) = -n^+ / sqrt(n^+ * n^-)

            = -sqrt(n^+ / n^-)
    
    Args:
        rewards: Tensor of shape [n,] containing binary rewards {0, 1}
        group_indices: List of indices for this group
        config: ContrastiveGRPOConfig
        device: Device for computation
    
    Returns:
        advantages: Tensor of shape [n,] with computed advantages
        metrics: Dictionary containing diagnostic information
    """
    n = len(group_indices)
    assert n == len(rewards), "Group indices and rewards must have same length"
    
    # Identify positive and negative samples
    positive_mask = (rewards > 0.5).float()  # r_i = 1
    negative_mask = (rewards <= 0.5).float()  # r_i = 0
    
    n_positive = positive_mask.sum().item()
    n_negative = negative_mask.sum().item()
    
    # Initialize advantages
    advantages = torch.zeros_like(rewards)
    
    # Handle degenerate cases
    metrics = {
        'n_total': n,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'is_degenerate': False
    }
    
    # Case 1: All rewards are identical (degenerate case)
    if n_positive == 0 or n_negative == 0:
        if config.handle_uniform_rewards:
            advantages.fill_(config.uniform_reward_advantage)
            metrics['is_degenerate'] = True
            metrics['degenerate_type'] = 'all_same'
            return advantages, metrics
        else:
            # Fall back to standard GRPO normalization
            r_mean = rewards.mean()
            r_std = rewards.std() + config.epsilon
            advantages = (rewards - r_mean) / r_std
            metrics['is_degenerate'] = True
            metrics['degenerate_type'] = 'all_same_fallback'
            return advantages, metrics
    
    # Case 2: Normal case - compute contrastive advantages
    # Mathematical formula:
    #   A^+ = sqrt(n^- / n^+)
    #   A^- = -sqrt(n^+ / n^-)
    
    advantage_positive = torch.sqrt(
        torch.tensor(n_negative / n_positive, device=device, dtype=rewards.dtype)
    )
    advantage_negative = -torch.sqrt(
        torch.tensor(n_positive / n_negative, device=device, dtype=rewards.dtype)
    )
    
    # Assign advantages
    advantages = positive_mask * advantage_positive + negative_mask * advantage_negative
    
    # Compute verification metrics
    if config.verify_equivalence:
        # Verify against standard GRPO computation
        r_mean = rewards.mean()
        r_std = rewards.std() + config.epsilon
        advantages_standard = (rewards - r_mean) / r_std
        
        max_diff = torch.abs(advantages - advantages_standard).max().item()
        metrics['verification_max_diff'] = max_diff
        metrics['verification_passed'] = (max_diff < 1e-5)
    
    # Log advantage statistics
    if config.log_advantage_stats:
        metrics.update({
            'adv_positive_value': advantage_positive.item(),
            'adv_negative_value': advantage_negative.item(),
            'adv_mean': advantages.mean().item(),
            'adv_std': advantages.std().item(),
            'adv_min': advantages.min().item(),
            'adv_max': advantages.max().item(),
        })
    
    # Log contrastive structure
    if config.log_contrastive_structure:
        # The contrastive weighting factor
        weight_ratio = torch.sqrt(torch.tensor(n_negative / n_positive, device=device))
        metrics['contrastive_weight_ratio'] = weight_ratio.item()
        
        # Balance of positive vs negative samples
        metrics['pos_neg_ratio'] = n_positive / n_negative if n_negative > 0 else float('inf')
        
        # Information content: how much signal do we have?
        # Higher variance in advantages -> more signal
        metrics['advantage_variance'] = advantages.var().item()
    
    return advantages, metrics


def format_grpo_as_contrastive(
    batch: DataProto,
    config: ContrastiveGRPOConfig,
    device: torch.device
) -> Tuple[DataProto, Dict[str, Any]]:
    """
    Reformulate GRPO advantages in explicit contrastive form.
    
    This function:
    1. Groups rollouts by UID (prompt)
    2. Partitions each group into positive/negative sets
    3. Computes contrastive advantages
    4. Maintains compatibility with existing GRPO trainer
    
    Mathematical Framework:
    -----------------------
    For each prompt q with rollouts {o_1, ..., o_n}:
    
        J_GRPO(q) = Σ_i A_i * log π_θ(o_i|q)
                  = A^+ * Σ_{j∈O^+} log π_θ(o_j|q) + A^- * Σ_{k∈O^-} log π_θ(o_k|q)
                  ∝ (1/n^+) * Σ_{j∈O^+} log π_θ(o_j|q) - (1/n^-) * Σ_{k∈O^-} log π_θ(o_k|q)
    
    This is the standard contrastive learning objective: maximize positive
    samples' log-probability while minimizing negative samples'.
    
    Args:
        batch: DataProto containing rollouts and rewards
        config: ContrastiveGRPOConfig
        device: Device for computation
    
    Returns:
        batch: Modified batch with contrastive advantages
        metrics: Aggregate metrics across all groups
    """
    # Extract UIDs and rewards
    uid = batch.non_tensor_batch["uid"]
    
    if "token_level_rewards" in batch.batch:
        seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1)  # [batch_size,]
    else:
        raise ValueError("token_level_rewards not found in batch")
    
    # Group by UID
    uid2indices = defaultdict(list)
    for idx, u in enumerate(uid):
        uid2indices[u].append(idx)
    
    # Initialize advantages tensor
    batch_size = len(uid)
    advantages = torch.zeros(batch_size, device=device, dtype=seq_rewards.dtype)
    
    # Compute returns for metrics (cumulative rewards)
    # Returns are needed for compute_data_metrics even though GRPO doesn't use them directly
    response_mask = batch.batch["response_mask"].bool()
    token_level_rewards = batch.batch["token_level_rewards"]
    # Compute returns as cumulative rewards (same shape as advantages for compatibility)
    returns = (token_level_rewards * response_mask).sum(dim=-1)  # [batch_size,]

    # Pre-compute sequence-level logprobs for actor/ref to form DPO-style ratios.
    # If ref logprob is missing, fall back to zeros so existing pipeline keeps running.
    actor_log_probs = batch.batch.get("log_probs", batch.batch.get("old_log_probs"))
    ref_log_probs = batch.batch.get("ref_log_prob")
    if actor_log_probs is not None:
        seq_actor_logp = (actor_log_probs * response_mask).sum(dim=-1)
    else:
        seq_actor_logp = None
    if ref_log_probs is not None:
        seq_ref_logp = (ref_log_probs * response_mask).sum(dim=-1)
    else:
        seq_ref_logp = torch.zeros_like(returns)
    
    # Aggregate metrics
    aggregate_metrics = {
        'total_groups': 0,
        'total_samples': 0,
        'total_positive': 0,
        'total_negative': 0,
        'degenerate_groups': 0,
        'groups_with_verification_issues': 0,
        'avg_group_size': 0.0,
        'avg_pos_neg_ratio': 0.0,
        'avg_advantage_variance': 0.0,
    }
    
    group_sizes = []
    pos_neg_ratios = []
    advantage_variances = []
    
    # Process each group
    for group_uid, group_indices in uid2indices.items():
        G = len(group_indices)
        group_sizes.append(G)
        
        if G < 2:
            # Single sample: no contrastive structure possible
            # Set advantage to 0 (no gradient signal)
            for idx in group_indices:
                advantages[idx] = 0.0
            continue
        
        # Extract group rewards
        group_rewards = seq_rewards[group_indices]

        # ------------------------------------------------------------
        # Original GRPO contrastive advantages (kept for reference):
        # group_advantages, group_metrics = compute_contrastive_advantages(
        #     rewards=group_rewards,
        #     group_indices=group_indices,
        #     config=config,
        #     device=device
        # )
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # Previous “mean-pos/mean-neg” DPO-ish margin (kept for reference):
        # logp_norm = group_actor_logp - torch.logsumexp(group_actor_logp, dim=0)
        # ref_norm = group_ref_logp - torch.logsumexp(group_ref_logp, dim=0)
        # delta = logp_norm - ref_norm
        # margin = beta * (delta_pos_mean - delta_neg_mean)
        # weight = sigmoid(-margin); adv_pos = weight*beta; adv_neg = -weight*beta
        # ------------------------------------------------------------

        # DPO-style pairwise formulation: for each (o+, o-) pair use -logσ(β·Δ_pair)
        # where Δ_pair = (logπ-logπ_ref)_pos - (logπ-logπ_ref)_neg.
        if seq_actor_logp is None:
            group_advantages = torch.zeros_like(group_rewards)
            group_metrics = {
                'n_positive': 0,
                'n_negative': 0,
                'is_degenerate': True,
                'degenerate_type': 'missing_log_probs',
            }
        else:
            group_actor_logp = seq_actor_logp[group_indices]
            group_ref_logp = seq_ref_logp[group_indices]

            pos_mask = (group_rewards > 0.5)
            neg_mask = ~pos_mask
            pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
            neg_idx = torch.nonzero(neg_mask, as_tuple=False).view(-1)
            n_pos = int(pos_mask.sum().item())
            n_neg = int(neg_mask.sum().item())

            group_metrics = {
                'n_positive': n_pos,
                'n_negative': n_neg,
                'is_degenerate': False,
            }

            if n_pos == 0 or n_neg == 0:
                group_advantages = torch.zeros_like(group_rewards)
                group_metrics['is_degenerate'] = True
                group_metrics['degenerate_type'] = 'no_positive_or_negative'
            else:
                # Group-normalize log-probs to avoid cross-group leakage
                logp_norm = group_actor_logp - torch.logsumexp(group_actor_logp, dim=0)
                ref_norm = group_ref_logp - torch.logsumexp(group_ref_logp, dim=0)
                delta = logp_norm - ref_norm  # per sample Δ = logπ - logπ_ref

                beta_val = getattr(config, "dpo_beta", 0.1)
                beta = torch.tensor(beta_val, device=device, dtype=group_rewards.dtype)

                group_advantages = torch.zeros_like(group_rewards)
                contrib_counts = torch.zeros_like(group_rewards)

                pair_margins = []
                for pi in pos_idx:
                    for ni in neg_idx:
                        delta_pair = delta[pi] - delta[ni]
                        pair_margins.append(delta_pair.item())
                        sig = torch.sigmoid(beta * delta_pair)
                        # loss = -logσ(beta * delta_pair); grad wrt delta_pair = sig - 1
                        grad_delta = sig - 1.0
                        pos_contrib = beta * grad_delta
                        neg_contrib = -beta * grad_delta

                        group_advantages[pi] += pos_contrib
                        group_advantages[ni] += neg_contrib
                        contrib_counts[pi] += 1
                        contrib_counts[ni] += 1

                # Average contributions per sample to keep scale similar across groups
                nonzero = contrib_counts > 0
                group_advantages[nonzero] = group_advantages[nonzero] / contrib_counts[nonzero]

                if pair_margins:
                    pair_margins_t = torch.tensor(pair_margins, device=device, dtype=group_rewards.dtype)
                    group_metrics.update({
                        'dpo_pair_margin_mean': pair_margins_t.mean().item(),
                        'dpo_pair_margin_std': pair_margins_t.std().item(),
                        'pos_neg_ratio': float(n_pos / n_neg) if n_neg > 0 else float('inf'),
                        'advantage_variance': group_advantages.var().item(),
                        'dpo_beta': float(beta_val),
                    })
        
        # Assign to batch
        for local_idx, global_idx in enumerate(group_indices):
            advantages[global_idx] = group_advantages[local_idx]
        
        # Update aggregate metrics
        aggregate_metrics['total_groups'] += 1
        aggregate_metrics['total_samples'] += G
        aggregate_metrics['total_positive'] += group_metrics['n_positive']
        aggregate_metrics['total_negative'] += group_metrics['n_negative']
        
        if group_metrics['is_degenerate']:
            aggregate_metrics['degenerate_groups'] += 1
        
        if config.verify_equivalence and not group_metrics.get('verification_passed', True):
            aggregate_metrics['groups_with_verification_issues'] += 1
        
        if 'pos_neg_ratio' in group_metrics and np.isfinite(group_metrics['pos_neg_ratio']):
            pos_neg_ratios.append(group_metrics['pos_neg_ratio'])
        
        if 'advantage_variance' in group_metrics:
            advantage_variances.append(group_metrics['advantage_variance'])
    
    # Compute averages
    if group_sizes:
        aggregate_metrics['avg_group_size'] = np.mean(group_sizes)
    if pos_neg_ratios:
        aggregate_metrics['avg_pos_neg_ratio'] = np.mean(pos_neg_ratios)
    if advantage_variances:
        aggregate_metrics['avg_advantage_variance'] = np.mean(advantage_variances)
    
    # Replace advantages in batch
    # In GRPO, advantages are typically stored at token level
    # We need to broadcast sequence-level advantages to token level
    
    if "response_mask" in batch.batch:
        response_mask = batch.batch["response_mask"]  # [batch_size, seq_len]
        seq_len = response_mask.shape[1]
        
        # Broadcast advantages to token level: [batch_size,] -> [batch_size, seq_len]
        token_advantages = advantages.unsqueeze(-1).expand(-1, seq_len)
        
        # Store advantages in batch (this will be used by actor update)
        batch.batch["advantages"] = token_advantages
        
        # Add returns to batch (token-level format for compatibility with compute_data_metrics)
        # Returns are cumulative rewards expanded to token level
        token_returns = returns.unsqueeze(-1).expand(-1, seq_len)
        batch.batch["returns"] = token_returns
    else:
        raise ValueError("response_mask not found in batch. Cannot broadcast advantages.")
    
    # Add diagnostic information
    batch.meta_info['contrastive_grpo_metrics'] = aggregate_metrics
    
    return batch, aggregate_metrics


def compute_contrastive_advantages_with_src(
    rewards: torch.Tensor,
    group_indices: List[int],
    config: ContrastiveGRPOSRCConfig,
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    计算Contrastive GRPO优势函数（与标准版本相同）
    
    数学公式：
    A+ = √(n-/n+) for r_i = 1
    A- = -√(n+/n-) for r_i = 0
    
    这部分与标准Contrastive GRPO完全相同
    
    Theoretical Assumption for GRPO-SRC:
    ------------------------------------
    We assume reward r(x,y) is outcome-based and invariant to conditioning context.
    
    Under this assumption:
    - Advantages are computed from rewards r(x,yw) and r(x,yl) under standard condition π(y|x)
    - These advantages are then used to weight log probabilities π(yw|x,yl) and π(yl|x,yw)
    - This is valid because: if yw is a good answer under π(yw|x), it should also be
      a good answer under π(yw|x,yl). The quality of the answer doesn't depend on
      the generation process/conditioning context.
    
    Example: For math problems, the correctness of an answer doesn't change whether
    the model saw a wrong answer before generating it. The reward is based solely
    on the (x, y) pair, not the generative process.
    
    If this assumption doesn't hold (e.g., reward depends on generation path),
    then we would need to re-evaluate rewards under conditioned distributions,
    which is typically infeasible.
    """
    n = len(group_indices)
    assert n == len(rewards), "Group indices and rewards must have same length"
    
    # 分区
    positive_mask = (rewards > 0.5).float()
    negative_mask = (rewards <= 0.5).float()
    
    n_positive = positive_mask.sum().item()
    n_negative = negative_mask.sum().item()
    
    advantages = torch.zeros_like(rewards)
    
    metrics = {
        'n_total': n,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'is_degenerate': False
    }
    
    # 退化情况
    if n_positive == 0 or n_negative == 0:
        if config.handle_uniform_rewards:
            advantages.fill_(config.uniform_reward_advantage)
            metrics['is_degenerate'] = True
            metrics['degenerate_type'] = 'all_same'
            return advantages, metrics
        else:
            r_mean = rewards.mean()
            r_std = rewards.std() + config.epsilon
            advantages = (rewards - r_mean) / r_std
            metrics['is_degenerate'] = True
            metrics['degenerate_type'] = 'all_same_fallback'
            return advantages, metrics
    
    # 计算对比优势
    advantage_positive = torch.sqrt(
        torch.tensor(n_negative / n_positive, device=device, dtype=rewards.dtype)
    )
    advantage_negative = -torch.sqrt(
        torch.tensor(n_positive / n_negative, device=device, dtype=rewards.dtype)
    )
    
    advantages = positive_mask * advantage_positive + negative_mask * advantage_negative
    
    # 验证
    if config.verify_equivalence:
        r_mean = rewards.mean()
        r_std = rewards.std() + config.epsilon
        advantages_standard = (rewards - r_mean) / r_std
        max_diff = torch.abs(advantages - advantages_standard).max().item()
        metrics['verification_max_diff'] = max_diff
        metrics['verification_passed'] = (max_diff < 1e-5)
    
    # 统计
    if config.log_advantage_stats:
        metrics.update({
            'adv_positive_value': advantage_positive.item(),
            'adv_negative_value': advantage_negative.item(),
            'adv_mean': advantages.mean().item(),
            'adv_std': advantages.std().item(),
        })
    
    return advantages, metrics


def create_self_reflective_pairs(
    batch: DataProto,
    config: ContrastiveGRPOSRCConfig,
    device: torch.device,
    uid2reward_std: Dict[str, float] = None,
    rollout_n: int = 2
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    创建self-reflective conditioning pairs
    
    对于每个组 (x, yw, yl):
    - 正向: (x ⊕ yl) → yw  (看到错误，生成正确)
    - 反向: (x ⊕ yw) → yl  (看到正确，识别错误)
    
    注意：当rollout_n=2时（数学假设的基础），只选择第一个正样本和第一个负样本进行配对，
    避免冗余信息。当rollout_n>2时，仍然遍历所有配对以充分利用信息。
    
    Args:
        rollout_n: Rollout数量，当为2时只创建一对配对（符合数学假设）
    
    Returns:
    - pairing_info: 配对信息和条件化的input_ids映射
    - pairing_metrics: 配对统计信息
    """
    uid = batch.non_tensor_batch["uid"]
    
    if "token_level_rewards" in batch.batch:
        seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1)  # [batch_size,]
    else:
        raise ValueError("token_level_rewards not found in batch")
    
    if "input_ids" not in batch.batch:
        raise ValueError("input_ids not found in batch")
    
    if "response_mask" not in batch.batch:
        raise ValueError("response_mask not in batch. Cannot create SRC pairs.")
    
    input_ids = batch.batch["input_ids"]  # [batch_size, seq_len]
    response_mask = batch.batch["response_mask"]  # [batch_size, seq_len]
    attention_mask = batch.batch.get("attention_mask", response_mask)  # [batch_size, seq_len]
    
    # 按UID分组
    uid2indices = defaultdict(list)
    for idx, u in enumerate(uid):
        uid2indices[u].append(idx)
    
    # 存储配对信息
    pairing_info = {
        'conditioned_input_ids': [],  # 条件化的input_ids
        'conditioned_attention_mask': [],  # 条件化的attention_mask
        'conditioned_response_mask': [],  # 条件化的response_mask
        'original_indices': [],  # 原始batch中的索引
        'conditioning_types': [],  # 'yw|yl' or 'yl|yw'
        'pair_advantages': [],  # 对应的advantages
    }
    
    pairing_metrics = {
        'total_pairs': 0,
        'positive_conditioned': 0,  # yw|yl pairs
        'negative_conditioned': 0,  # yl|yw pairs
        'groups_processed': 0,
        'groups_skipped': 0,
    }
    
    # 获取tokenizer的sep token（需要从配置或模型中获取）
    # 这里使用一个占位符，实际使用时需要从tokenizer获取
    sep_token_id = getattr(config, 'sep_token_id', None)
    if sep_token_id is None:
        # 尝试从batch或配置中获取，如果没有则使用一个特殊值
        # 注意：这需要在配置中指定或从tokenizer获取
        sep_token_id = 2  # 常见的是2，但应该从实际tokenizer获取
    
    for group_uid, group_indices in uid2indices.items():
        if len(group_indices) < 2:
            pairing_metrics['groups_skipped'] += 1
            continue
        
        # 检查该group的reward std，如果std=0则跳过SRC配对（使用标准GRPO）
        if uid2reward_std is not None:
            reward_std = uid2reward_std.get(group_uid, None)
            if reward_std is not None and reward_std == 0:
                # std=0，跳过SRC配对，使用标准GRPO
                pairing_metrics['groups_skipped'] += 1
                continue
        
        pairing_metrics['groups_processed'] += 1
        
        # 获取组内rewards
        group_rewards = seq_rewards[group_indices].cpu()
        
        # 找到正负样本
        pos_mask = group_rewards > 0.5
        neg_mask = group_rewards <= 0.5
        
        pos_indices = [group_indices[i] for i in range(len(group_indices)) if pos_mask[i]]
        neg_indices = [group_indices[i] for i in range(len(group_indices)) if neg_mask[i]]
        
        if not pos_indices or not neg_indices:
            pairing_metrics['groups_skipped'] += 1
            continue
        
        # 计算advantages（用于配对）
        group_advantages, adv_metrics = compute_contrastive_advantages_with_src(
            rewards=seq_rewards[group_indices],
            group_indices=group_indices,
            config=config,
            device=device
        )
        
        # 获取明确的positive和negative advantage值
        # A_w = √(n-/n+) > 0 用于正向配对 (yw|yl)
        # A_l = -√(n+/n-) < 0 用于反向配对 (yl|yw)
        advantage_positive = adv_metrics.get('adv_positive_value', None)
        advantage_negative = adv_metrics.get('adv_negative_value', None)
        
        # 如果metrics中没有，直接从group_advantages计算
        if advantage_positive is None or advantage_negative is None:
            n_pos = len(pos_indices)
            n_neg = len(neg_indices)
            if n_pos > 0 and n_neg > 0:
                advantage_positive = torch.sqrt(torch.tensor(n_neg / n_pos, device=device, dtype=seq_rewards.dtype))
                advantage_negative = -torch.sqrt(torch.tensor(n_pos / n_neg, device=device, dtype=seq_rewards.dtype))
            else:
                # 退化情况：使用group_advantages的平均值
                pos_advs = [group_advantages[group_indices.index(idx)] for idx in pos_indices if idx in group_indices]
                neg_advs = [group_advantages[group_indices.index(idx)] for idx in neg_indices if idx in group_indices]
                advantage_positive = torch.tensor(sum(pos_advs) / len(pos_advs), device=device) if pos_advs else torch.tensor(0.0, device=device)
                advantage_negative = torch.tensor(sum(neg_advs) / len(neg_advs), device=device) if neg_advs else torch.tensor(0.0, device=device)
        
        # 为每个正样本创建条件（yw | x, yl）
        # 当rollout_n=2时，只选择第一个正样本和第一个负样本（符合数学假设，避免冗余）
        # 当rollout_n>2时，遍历所有配对以充分利用信息
        # 使用 advantage_positive (A_w > 0) 来加权正向配对
        if rollout_n == 2:
            # 数学假设的基础：只选择第一个正样本和第一个负样本
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_idx = pos_indices[0]
                neg_idx = neg_indices[0]
                # 创建单个配对（避免冗余）
                pos_seq = input_ids[pos_idx]
                pos_attn = attention_mask[pos_idx]
                pos_resp_mask = response_mask[pos_idx]
                
                prompt_end = pos_resp_mask.argmax().item() if pos_resp_mask.sum() > 0 else 0
                prompt = pos_seq[:prompt_end]
                pos_response = pos_seq[prompt_end:]
                
                neg_seq = input_ids[neg_idx]
                neg_resp_mask = response_mask[neg_idx]
                neg_resp_end = neg_resp_mask.argmax().item() + neg_resp_mask.sum().int().item() if neg_resp_mask.sum() > 0 else len(neg_seq)
                neg_resp_start = neg_resp_mask.argmax().item() if neg_resp_mask.sum() > 0 else 0
                neg_response = neg_seq[neg_resp_start:neg_resp_end]
                
                # 创建条件化序列: prompt + sep + neg_response + sep + pos_response
                conditioned_seq = torch.cat([
                    prompt,
                    torch.tensor([sep_token_id], device=device, dtype=prompt.dtype),
                    neg_response,
                    torch.tensor([sep_token_id], device=device, dtype=prompt.dtype),
                    pos_response
                ])
                
                seq_len = len(conditioned_seq)
                conditioned_attn = torch.ones(seq_len, device=device, dtype=attention_mask.dtype)
                conditioned_resp_mask = torch.zeros(seq_len, device=device, dtype=response_mask.dtype)
                resp_start = len(prompt) + 1 + len(neg_response) + 1
                conditioned_resp_mask[resp_start:] = 1
                
                # 填充或截断
                max_seq_len = input_ids.shape[1]
                max_conditioned_len = min(max_seq_len, 4096)
                if len(conditioned_seq) > max_conditioned_len:
                    conditioned_seq = conditioned_seq[:max_conditioned_len]
                    conditioned_attn = conditioned_attn[:max_conditioned_len]
                    conditioned_resp_mask = conditioned_resp_mask[:max_conditioned_len]
                elif len(conditioned_seq) < max_conditioned_len:
                    pad_len = max_conditioned_len - len(conditioned_seq)
                    pad_token_id = getattr(config, 'pad_token_id', 0)
                    conditioned_seq = torch.cat([
                        conditioned_seq,
                        torch.full((pad_len,), pad_token_id, device=device, dtype=conditioned_seq.dtype)
                    ])
                    conditioned_attn = torch.cat([
                        conditioned_attn,
                        torch.zeros(pad_len, device=device, dtype=conditioned_attn.dtype)
                    ])
                    conditioned_resp_mask = torch.cat([
                        conditioned_resp_mask,
                        torch.zeros(pad_len, device=device, dtype=conditioned_resp_mask.dtype)
                    ])
                
                pairing_info['conditioned_input_ids'].append(conditioned_seq)
                pairing_info['conditioned_attention_mask'].append(conditioned_attn)
                pairing_info['conditioned_response_mask'].append(conditioned_resp_mask)
                pairing_info['original_indices'].append(pos_idx)
                pairing_info['conditioning_types'].append('yw|yl')
                if isinstance(advantage_positive, torch.Tensor):
                    pairing_info['pair_advantages'].append(advantage_positive.item())
                else:
                    pairing_info['pair_advantages'].append(advantage_positive)
                pairing_metrics['positive_conditioned'] += 1
        else:
            # rollout_n > 2: 遍历所有配对以充分利用信息
            for pos_idx in pos_indices:
                for neg_idx in neg_indices:
                    # 提取prompt部分（response_mask之前的部分）
                    pos_seq = input_ids[pos_idx]
                    pos_attn = attention_mask[pos_idx]
                    pos_resp_mask = response_mask[pos_idx]
                    
                    # 找到prompt的结束位置（response开始之前）
                    prompt_end = pos_resp_mask.argmax().item() if pos_resp_mask.sum() > 0 else 0
                    
                    prompt = pos_seq[:prompt_end]
                    pos_response = pos_seq[prompt_end:]
                    
                    # 获取负样本的response
                    neg_seq = input_ids[neg_idx]
                    neg_resp_mask = response_mask[neg_idx]
                    neg_resp_end = neg_resp_mask.argmax().item() + neg_resp_mask.sum().int().item() if neg_resp_mask.sum() > 0 else len(neg_seq)
                    neg_resp_start = neg_resp_mask.argmax().item() if neg_resp_mask.sum() > 0 else 0
                    neg_response = neg_seq[neg_resp_start:neg_resp_end]
                    
                    # 创建条件化序列: prompt + sep + neg_response + sep + pos_response
                    conditioned_seq = torch.cat([
                        prompt,
                        torch.tensor([sep_token_id], device=device, dtype=prompt.dtype),
                        neg_response,
                        torch.tensor([sep_token_id], device=device, dtype=prompt.dtype),
                        pos_response
                    ])
                    
                    # 创建对应的attention_mask和response_mask
                    seq_len = len(conditioned_seq)
                    conditioned_attn = torch.ones(seq_len, device=device, dtype=attention_mask.dtype)
                    
                    # response_mask只覆盖最后的pos_response部分
                    conditioned_resp_mask = torch.zeros(seq_len, device=device, dtype=response_mask.dtype)
                    resp_start = len(prompt) + 1 + len(neg_response) + 1  # 跳过prompt + sep + neg_response + sep
                    conditioned_resp_mask[resp_start:] = 1
                    
                    # 填充或截断到统一长度（如果需要）
                    # Use a more conservative limit to avoid exceeding log_prob_max_token_len_per_gpu
                    # Conditioned sequences contain two responses, so we need to cap at a reasonable length
                    max_seq_len = input_ids.shape[1]
                    # Limit to 4096 to match log_prob_max_token_len_per_gpu configuration (reduced for memory efficiency)
                    max_conditioned_len = min(max_seq_len, 4096)  # Allow up to 4096 for conditioned sequences
                    if len(conditioned_seq) > max_conditioned_len:
                        conditioned_seq = conditioned_seq[:max_conditioned_len]
                        conditioned_attn = conditioned_attn[:max_conditioned_len]
                        conditioned_resp_mask = conditioned_resp_mask[:max_conditioned_len]
                    elif len(conditioned_seq) < max_conditioned_len:
                        # Pad to max_conditioned_len (not max_seq_len) to keep consistent with truncation
                        pad_len = max_conditioned_len - len(conditioned_seq)
                        pad_token_id = getattr(config, 'pad_token_id', 0)
                        conditioned_seq = torch.cat([
                            conditioned_seq,
                            torch.full((pad_len,), pad_token_id, device=device, dtype=conditioned_seq.dtype)
                        ])
                        conditioned_attn = torch.cat([
                            conditioned_attn,
                            torch.zeros(pad_len, device=device, dtype=conditioned_attn.dtype)
                        ])
                        conditioned_resp_mask = torch.cat([
                            conditioned_resp_mask,
                            torch.zeros(pad_len, device=device, dtype=conditioned_resp_mask.dtype)
                        ])
                    
                    pairing_info['conditioned_input_ids'].append(conditioned_seq)
                    pairing_info['conditioned_attention_mask'].append(conditioned_attn)
                    pairing_info['conditioned_response_mask'].append(conditioned_resp_mask)
                    pairing_info['original_indices'].append(pos_idx)
                    pairing_info['conditioning_types'].append('yw|yl')
                    # 正向配对使用 advantage_positive (A_w > 0): 鼓励 π(yw|x,yl)
                    # 注意：advantages的平均处理在format_grpo_as_contrastive_with_src中进行
                    # 这里只记录原始的advantage值（用于日志记录）
                    if isinstance(advantage_positive, torch.Tensor):
                        pairing_info['pair_advantages'].append(advantage_positive.item())
                    else:
                        pairing_info['pair_advantages'].append(advantage_positive)
                    pairing_metrics['positive_conditioned'] += 1
        
        # 对称：为每个负样本创建条件（yl | x, yw）
        # 当rollout_n=2时，只选择第一个正样本和第一个负样本（符合数学假设，避免冗余）
        # 当rollout_n>2时，遍历所有配对以充分利用信息
        if config.symmetric_conditioning:
            if rollout_n == 2:
                # 数学假设的基础：只选择第一个正样本和第一个负样本
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    pos_idx = pos_indices[0]
                    neg_idx = neg_indices[0]
                    # 创建单个配对（避免冗余）
                    neg_seq = input_ids[neg_idx]
                    neg_attn = attention_mask[neg_idx]
                    neg_resp_mask = response_mask[neg_idx]
                    
                    neg_resp_start = neg_resp_mask.argmax().item() if neg_resp_mask.sum() > 0 else 0
                    prompt = neg_seq[:neg_resp_start]
                    neg_response = neg_seq[neg_resp_start:]
                    
                    pos_seq = input_ids[pos_idx]
                    pos_resp_mask = response_mask[pos_idx]
                    pos_resp_start = pos_resp_mask.argmax().item() if pos_resp_mask.sum() > 0 else 0
                    pos_response = pos_seq[pos_resp_start:]
                    
                    # 创建条件化序列: prompt + sep + pos_response + sep + neg_response
                    conditioned_seq = torch.cat([
                        prompt,
                        torch.tensor([sep_token_id], device=device, dtype=prompt.dtype),
                        pos_response,
                        torch.tensor([sep_token_id], device=device, dtype=prompt.dtype),
                        neg_response
                    ])
                    
                    seq_len = len(conditioned_seq)
                    conditioned_attn = torch.ones(seq_len, device=device, dtype=attention_mask.dtype)
                    conditioned_resp_mask = torch.zeros(seq_len, device=device, dtype=response_mask.dtype)
                    resp_start = len(prompt) + 1 + len(pos_response) + 1
                    conditioned_resp_mask[resp_start:] = 1
                    
                    # 填充或截断
                    max_seq_len = input_ids.shape[1]
                    max_conditioned_len = min(max_seq_len, 4096)
                    if len(conditioned_seq) > max_conditioned_len:
                        conditioned_seq = conditioned_seq[:max_conditioned_len]
                        conditioned_attn = conditioned_attn[:max_conditioned_len]
                        conditioned_resp_mask = conditioned_resp_mask[:max_conditioned_len]
                    elif len(conditioned_seq) < max_conditioned_len:
                        pad_len = max_conditioned_len - len(conditioned_seq)
                        pad_token_id = getattr(config, 'pad_token_id', 0)
                        conditioned_seq = torch.cat([
                            conditioned_seq,
                            torch.full((pad_len,), pad_token_id, device=device, dtype=conditioned_seq.dtype)
                        ])
                        conditioned_attn = torch.cat([
                            conditioned_attn,
                            torch.zeros(pad_len, device=device, dtype=conditioned_attn.dtype)
                        ])
                        conditioned_resp_mask = torch.cat([
                            conditioned_resp_mask,
                            torch.zeros(pad_len, device=device, dtype=conditioned_resp_mask.dtype)
                        ])
                    
                    pairing_info['conditioned_input_ids'].append(conditioned_seq)
                    pairing_info['conditioned_attention_mask'].append(conditioned_attn)
                    pairing_info['conditioned_response_mask'].append(conditioned_resp_mask)
                    pairing_info['original_indices'].append(neg_idx)
                    pairing_info['conditioning_types'].append('yl|yw')
                    if isinstance(advantage_negative, torch.Tensor):
                        pairing_info['pair_advantages'].append(advantage_negative.item())
                    else:
                        pairing_info['pair_advantages'].append(advantage_negative)
                    pairing_metrics['negative_conditioned'] += 1
            else:
                # rollout_n > 2: 遍历所有配对以充分利用信息
                for neg_idx in neg_indices:
                    for pos_idx in pos_indices:
                        # 提取prompt和responses
                        neg_seq = input_ids[neg_idx]
                        neg_attn = attention_mask[neg_idx]
                        neg_resp_mask = response_mask[neg_idx]
                        
                        neg_resp_start = neg_resp_mask.argmax().item() if neg_resp_mask.sum() > 0 else 0
                        prompt = neg_seq[:neg_resp_start]
                        neg_response = neg_seq[neg_resp_start:]
                        
                        pos_seq = input_ids[pos_idx]
                        pos_resp_mask = response_mask[pos_idx]
                        pos_resp_start = pos_resp_mask.argmax().item() if pos_resp_mask.sum() > 0 else 0
                        pos_response = pos_seq[pos_resp_start:]
                        
                        # 创建条件化序列: prompt + sep + pos_response + sep + neg_response
                        conditioned_seq = torch.cat([
                            prompt,
                            torch.tensor([sep_token_id], device=device, dtype=prompt.dtype),
                            pos_response,
                            torch.tensor([sep_token_id], device=device, dtype=prompt.dtype),
                            neg_response
                        ])
                        
                        seq_len = len(conditioned_seq)
                        conditioned_attn = torch.ones(seq_len, device=device, dtype=attention_mask.dtype)
                        conditioned_resp_mask = torch.zeros(seq_len, device=device, dtype=response_mask.dtype)
                        resp_start = len(prompt) + 1 + len(pos_response) + 1
                        conditioned_resp_mask[resp_start:] = 1
                        
                        # 填充或截断
                        # Use a more conservative limit to avoid exceeding log_prob_max_token_len_per_gpu
                        max_seq_len = input_ids.shape[1]
                        max_conditioned_len = min(max_seq_len, 4096)  # Allow up to 4096 for conditioned sequences
                        if len(conditioned_seq) > max_conditioned_len:
                            conditioned_seq = conditioned_seq[:max_conditioned_len]
                            conditioned_attn = conditioned_attn[:max_conditioned_len]
                            conditioned_resp_mask = conditioned_resp_mask[:max_conditioned_len]
                        elif len(conditioned_seq) < max_conditioned_len:
                            # Pad to max_conditioned_len (not max_seq_len) to keep consistent with truncation
                            pad_len = max_conditioned_len - len(conditioned_seq)
                            pad_token_id = getattr(config, 'pad_token_id', 0)
                            conditioned_seq = torch.cat([
                                conditioned_seq,
                                torch.full((pad_len,), pad_token_id, device=device, dtype=conditioned_seq.dtype)
                            ])
                            conditioned_attn = torch.cat([
                                conditioned_attn,
                                torch.zeros(pad_len, device=device, dtype=conditioned_attn.dtype)
                            ])
                            conditioned_resp_mask = torch.cat([
                                conditioned_resp_mask,
                                torch.zeros(pad_len, device=device, dtype=conditioned_resp_mask.dtype)
                            ])
                        
                        pairing_info['conditioned_input_ids'].append(conditioned_seq)
                        pairing_info['conditioned_attention_mask'].append(conditioned_attn)
                        pairing_info['conditioned_response_mask'].append(conditioned_resp_mask)
                        pairing_info['original_indices'].append(neg_idx)
                        pairing_info['conditioning_types'].append('yl|yw')
                        # 反向配对使用 advantage_negative (A_l < 0): 抑制 π(yl|x,yw)
                        # 注意：advantages的平均处理在format_grpo_as_contrastive_with_src中进行
                        # 这里只记录原始的advantage值（用于日志记录）
                        if isinstance(advantage_negative, torch.Tensor):
                            pairing_info['pair_advantages'].append(advantage_negative.item())
                        else:
                            pairing_info['pair_advantages'].append(advantage_negative)
                        pairing_metrics['negative_conditioned'] += 1
    
    pairing_metrics['total_pairs'] = len(pairing_info['original_indices'])
    
    # 转换为tensors
    if pairing_info['conditioned_input_ids']:
        pairing_info['conditioned_input_ids'] = torch.stack(pairing_info['conditioned_input_ids'])
        pairing_info['conditioned_attention_mask'] = torch.stack(pairing_info['conditioned_attention_mask'])
        pairing_info['conditioned_response_mask'] = torch.stack(pairing_info['conditioned_response_mask'])
        pairing_info['original_indices'] = torch.tensor(pairing_info['original_indices'], device=device, dtype=torch.long)
        # pair_advantages is a list of Python floats, convert to tensor
        pairing_info['pair_advantages'] = torch.tensor(pairing_info['pair_advantages'], device=device, dtype=seq_rewards.dtype)
    else:
        # 如果没有配对，返回空tensors
        pairing_info['conditioned_input_ids'] = torch.empty((0, input_ids.shape[1]), device=device, dtype=input_ids.dtype)
        pairing_info['conditioned_attention_mask'] = torch.empty((0, input_ids.shape[1]), device=device, dtype=attention_mask.dtype)
        pairing_info['conditioned_response_mask'] = torch.empty((0, input_ids.shape[1]), device=device, dtype=response_mask.dtype)
        pairing_info['original_indices'] = torch.empty((0,), device=device, dtype=torch.long)
        pairing_info['pair_advantages'] = torch.empty((0,), device=device, dtype=seq_rewards.dtype)
    
    return pairing_info, pairing_metrics


def format_grpo_as_contrastive_with_src(
    batch: DataProto,
    config: ContrastiveGRPOSRCConfig,
    device: torch.device
) -> Tuple[DataProto, Dict[str, Any]]:
    """
    将GRPO重塑为Contrastive GRPO with Self-Reflective Conditioning格式
    
    核心思想：
    1. 使用Contrastive GRPO的优势函数
    2. 如果启用SRC，创建条件化的输入对
    3. 在后续的log_prob计算中，使用条件化的输入
    
    Returns:
    - batch: 修改后的batch，包含条件化的输入（如果启用SRC）
    - metrics: 统计信息
    """
    uid = batch.non_tensor_batch["uid"]
    
    if "token_level_rewards" in batch.batch:
        seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1)  # [batch_size,]
    else:
        raise ValueError("token_level_rewards not found in batch")
    
    # 按UID分组
    uid2indices = defaultdict(list)
    for idx, u in enumerate(uid):
        uid2indices[u].append(idx)
    
    # 计算advantages（与标准contrastive GRPO相同）
    batch_size = len(uid)
    advantages = torch.zeros(batch_size, device=device, dtype=seq_rewards.dtype)
    
    # Compute returns for metrics (cumulative rewards)
    # Returns are needed for compute_data_metrics even though GRPO doesn't use them directly
    response_mask = batch.batch["response_mask"].bool()
    token_level_rewards = batch.batch["token_level_rewards"]
    # Compute returns as cumulative rewards (same shape as advantages for compatibility)
    returns = (token_level_rewards * response_mask).sum(dim=-1)  # [batch_size,]
    
    aggregate_metrics = {
        'total_groups': 0,
        'total_samples': 0,
        'total_positive': 0,
        'total_negative': 0,
        'degenerate_groups': 0,
        'avg_group_size': 0.0,
        'avg_pos_neg_ratio': 0.0,
    }
    
    group_sizes = []
    pos_neg_ratios = []
    
    # 处理每个组，计算advantages
    for group_uid, group_indices in uid2indices.items():
        G = len(group_indices)
        group_sizes.append(G)
        
        if G < 2:
            for idx in group_indices:
                advantages[idx] = 0.0
            continue
        
        group_rewards = seq_rewards[group_indices]
        group_advantages, group_metrics = compute_contrastive_advantages_with_src(
            rewards=group_rewards,
            group_indices=group_indices,
            config=config,
            device=device
        )
        
        for local_idx, global_idx in enumerate(group_indices):
            advantages[global_idx] = group_advantages[local_idx]
        
        aggregate_metrics['total_groups'] += 1
        aggregate_metrics['total_samples'] += G
        aggregate_metrics['total_positive'] += group_metrics['n_positive']
        aggregate_metrics['total_negative'] += group_metrics['n_negative']
        
        if group_metrics['is_degenerate']:
            aggregate_metrics['degenerate_groups'] += 1
        
        if 'pos_neg_ratio' in group_metrics and np.isfinite(group_metrics.get('pos_neg_ratio', float('inf'))):
            pos_neg_ratios.append(group_metrics['pos_neg_ratio'])
    
    if group_sizes:
        aggregate_metrics['avg_group_size'] = np.mean(group_sizes)
    if pos_neg_ratios:
        aggregate_metrics['avg_pos_neg_ratio'] = np.mean(pos_neg_ratios)
    
    # 如果启用SRC，创建self-reflective pairs
    # 但只对std>0的group使用SRC，std=0的group使用标准GRPO
    src_metrics = {}
    if config.use_self_reflection:
        # 计算每个group的reward std，用于判断是否使用SRC
        uid2reward_std = {}
        uid2reward_mean = {}
        std_zero_all_one_count = 0  # std=0且全为1的数量
        std_zero_all_zero_count = 0  # std=0且全为0的数量
        
        for group_uid, group_indices in uid2indices.items():
            if len(group_indices) < 2:
                continue
            group_rewards = seq_rewards[group_indices].cpu().numpy()
            reward_std = np.std(group_rewards)
            reward_mean = np.mean(group_rewards)
            uid2reward_std[group_uid] = reward_std
            uid2reward_mean[group_uid] = reward_mean
            
            # 记录std=0时的统计信息
            if reward_std == 0:
                if reward_mean > 0.5:  # 全为1
                    std_zero_all_one_count += 1
                else:  # 全为0
                    std_zero_all_zero_count += 1
        
        # 只对std>0的group创建SRC配对
        # 获取rollout数量（从batch的meta_info或配置中获取）
        # 注意：这里需要从外部传入rollout_n，因为batch中可能没有这个信息
        # 默认使用2（符合数学假设）
        rollout_n = getattr(config, 'rollout_n', 2)
        pairing_info, pairing_metrics = create_self_reflective_pairs(
            batch=batch,
            config=config,
            device=device,
            uid2reward_std=uid2reward_std,  # 传入std信息，用于过滤
            rollout_n=rollout_n  # 传入rollout数量
        )
        
        # 将配对信息存储到batch.meta_info中，供后续使用
        batch.meta_info['src_pairing_info'] = pairing_info
        
        src_metrics = {
            'src/total_pairs': pairing_metrics['total_pairs'],
            'src/positive_conditioned': pairing_metrics['positive_conditioned'],
            'src/negative_conditioned': pairing_metrics['negative_conditioned'],
            'src/groups_processed': pairing_metrics['groups_processed'],
            'src/groups_skipped': pairing_metrics['groups_skipped'],
            'src/std_zero_groups': std_zero_all_one_count + std_zero_all_zero_count,
            'src/std_zero_all_one': std_zero_all_one_count,  # std=0且全为1的数量
            'src/std_zero_all_zero': std_zero_all_zero_count,  # std=0且全为0的数量
        }
    
    # For GRPO-SRC: We keep the original batch structure, so advantages are computed on the original batch.
    # No mapping needed - the batch structure remains the same as standard GRPO.
    
    # 广播advantages到token level
    if "response_mask" in batch.batch:
        response_mask = batch.batch["response_mask"]
        seq_len = response_mask.shape[1]
        token_advantages = advantages.unsqueeze(-1).expand(-1, seq_len)
        batch.batch["advantages"] = token_advantages
        
        # Add returns to batch (token-level format for compatibility with compute_data_metrics)
        # Returns are cumulative rewards expanded to token level
        token_returns = returns.unsqueeze(-1).expand(-1, seq_len)
        batch.batch["returns"] = token_returns
    else:
        raise ValueError("response_mask not found in batch. Cannot broadcast advantages.")
    
    # 合并metrics
    aggregate_metrics.update(src_metrics)
    
    return batch, aggregate_metrics


# Note: GRPO-SRC (Contrastive GRPO with Self-Reflective Conditioning) implementation
# is available but not integrated into RefineDAPOTrainer. The following functions
# are provided for potential future use:
# - ContrastiveGRPOSRCConfig
# - compute_contrastive_advantages_with_src
# - create_self_reflective_pairs
# - format_grpo_as_contrastive_with_src
# - analyze_contrastive_structure (diagnostic utility, not currently used)

