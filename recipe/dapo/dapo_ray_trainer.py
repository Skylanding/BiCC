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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

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
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _build_oracle0_prompt_batch_from_pred(
        self, batch: DataProto
    ) -> tuple[DataProto, torch.Tensor] | None:
        """
        Build an "oracle R=0" prompt batch for OPPO.

        Oracle knows the answer but R splits into two classes:
        - R=1: correct
        - R=0: incorrect (condition on a negative answer from the same uid)

        This uses `pred` (and optionally `acc`) from reward extra info to pick a wrong answer per prompt uid.
        If no negative answer is available, returns None (caller will fall back to std/old_log_probs).
        """
        # Need prompts/responses/masks
        if "prompts" not in batch.batch or "responses" not in batch.batch:
            return None
        if "response_mask" not in batch.batch:
            return None
        if "pred" not in batch.non_tensor_batch:
            return None
        if "uid" not in batch.non_tensor_batch:
            return None
        if "reward_model" not in batch.non_tensor_batch:
            return None

        # We strongly prefer using acc to find truly negative trajectories
        acc_arr = batch.non_tensor_batch.get("acc", None)
        pred_arr = batch.non_tensor_batch.get("pred", None)
        uid_arr = batch.non_tensor_batch.get("uid", None)
        if pred_arr is None or uid_arr is None:
            return None

        # Build: uid -> a negative answer string (only from acc==0 within the same uid)
        uid2neg: dict[str, str] = {}

        def _norm_str(x) -> str | None:
            if x is None:
                return None
            s = str(x).strip()
            return s if s else None

        # First pass: collect negatives
        for i in range(len(uid_arr)):
            uid = str(uid_arr[i])
            pred_s = _norm_str(pred_arr[i])
            if pred_s is None:
                continue
            # Avoid picking predictions that match ground truth
            gt_s = _norm_str(
                batch[i].non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            )
            if gt_s is not None and pred_s == gt_s:
                continue
            is_neg = None
            if acc_arr is not None:
                try:
                    is_neg = (bool(acc_arr[i]) is False)
                except Exception:
                    is_neg = None
            # If we know it's negative, store as uid's negative answer
            if is_neg is True and uid not in uid2neg:
                uid2neg[uid] = pred_s

        if not uid2neg:
            return None

        # Second pass: choose a negative answer for each trajectory
        neg_answers: list[str] = []
        valid_mask: list[bool] = []
        for i in range(len(uid_arr)):
            uid = str(uid_arr[i])
            chosen = uid2neg.get(uid, None)
            # Avoid choosing predictions that match ground truth
            gt_s = _norm_str(
                batch[i].non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            )
            if chosen is not None and gt_s is not None and chosen == gt_s:
                chosen = None
            if chosen is None:
                chosen = ""
                valid_mask.append(False)
            else:
                valid_mask.append(True)
            neg_answers.append(chosen)

        if not any(valid_mask):
            return None

        oracle0_batch = self._build_oracle_prompt_batch_with_answers(batch=batch, answers=neg_answers)
        if oracle0_batch is None:
            return None
        valid_tensor = torch.tensor(valid_mask, device=oracle0_batch.batch["responses"].device)
        return oracle0_batch, valid_tensor

    def _build_oracle_prompt_batch_with_answers(self, batch: DataProto, answers: list[str]) -> DataProto | None:
        """
        Internal helper to construct oracle-augmented prompts with a per-sample answer string.

        This mirrors `RayPPOTrainer._build_oracle_prompt_batch` but takes `answers` explicitly,
        allowing OPPO to build R=1 and R=0 oracle conditions separately.
        """
        if "prompts" not in batch.batch or "responses" not in batch.batch:
            return None
        if "response_mask" not in batch.batch:
            return None
        if len(answers) != len(batch):
            return None

        prompt_ids = batch.batch["prompts"]
        response_ids = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        response_mask = batch.batch["response_mask"]

        batch_size, prompt_len = prompt_ids.shape
        response_len = response_ids.shape[1]
        prompt_attention_mask = attention_mask[:, :prompt_len]
        response_attention_mask = attention_mask[:, -response_len:]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        oppo_cfg = self.config.algorithm.oppo
        truncation = oppo_cfg.get("truncation", None) or self.config.data.get("truncation", "left")

        new_prompt_ids = torch.full_like(prompt_ids, fill_value=pad_token_id)
        new_prompt_attention_mask = torch.zeros_like(prompt_attention_mask)

        oracle_prefix = oppo_cfg.get("oracle_prompt_prefix", "")
        oracle_suffix = oppo_cfg.get("oracle_prompt_suffix", "")
        oracle_sep = oppo_cfg.get("oracle_prompt_sep", " ")

        for i in range(batch_size):
            ans = str(answers[i]) if answers[i] is not None else ""
            ans = ans.strip()
            oracle_text = ""
            if ans:
                oracle_text = f"{oracle_prefix}{ans}{oracle_suffix}"
                if oracle_sep and oracle_text:
                    oracle_text = f"{oracle_sep}{oracle_text}"

            valid_prompt_ids = prompt_ids[i][prompt_attention_mask[i].bool()]
            oracle_suffix_ids = self.tokenizer.encode(oracle_text, add_special_tokens=False) if oracle_text else []

            merged_ids = torch.tensor(
                list(valid_prompt_ids.tolist()) + list(oracle_suffix_ids),
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )

            if merged_ids.numel() > prompt_len:
                if truncation == "left":
                    merged_ids = merged_ids[-prompt_len:]
                elif truncation == "right":
                    merged_ids = merged_ids[:prompt_len]
                elif truncation == "middle":
                    left_half = prompt_len // 2
                    right_half = prompt_len - left_half
                    merged_ids = torch.cat([merged_ids[:left_half], merged_ids[-right_half:]], dim=0)
                elif truncation == "error":
                    raise RuntimeError(f"Oracle prompt length {merged_ids.numel()} exceeds max {prompt_len}.")
                else:
                    merged_ids = merged_ids[-prompt_len:]

            pad_len = prompt_len - merged_ids.numel()
            if merged_ids.numel() > 0:
                new_prompt_ids[i, pad_len:] = merged_ids
                new_prompt_attention_mask[i, pad_len:] = 1

        new_attention_mask = torch.cat([new_prompt_attention_mask, response_attention_mask], dim=-1)
        input_ids = torch.cat([new_prompt_ids, response_ids], dim=-1)
        from verl.trainer.ppo.ray_trainer import compute_position_id_with_mask

        position_ids = compute_position_id_with_mask(new_attention_mask)

        return DataProto.from_dict(
            tensors={
                "prompts": new_prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": new_attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
            },
            meta_info=deepcopy(batch.meta_info),
        )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
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
        # currently, we only support validation using the reward_function.
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
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
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
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                        # For OPPO: Set success scores for R_bar calculation
                        # Use acc if available, otherwise normalize token-level rewards sum to [0, 1]
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.OPPO:
                            if "acc" in new_batch.non_tensor_batch:
                                # Use binary accuracy if available (preferred for OPPO)
                                success_scores = torch.tensor(
                                    new_batch.non_tensor_batch["acc"],
                                    dtype=new_batch.batch["token_level_rewards"].dtype,
                                    device=new_batch.batch["token_level_rewards"].device,
                                )
                            else:
                                # Fallback: normalize token-level rewards sum to [0, 1]
                                # Assuming rewards are typically in range [-2, 1] for DAPO math tasks
                                reward_sum = new_batch.batch["token_level_rewards"].sum(dim=-1)
                                # Normalize: (reward_sum - min) / (max - min)
                                # For DAPO math: typical range is [-2, 1], normalize to [0, 1]
                                reward_min = reward_sum.min()
                                reward_max = reward_sum.max()
                                if reward_max > reward_min:
                                    success_scores = (reward_sum - reward_min) / (reward_max - reward_min)
                                else:
                                    # If all rewards are the same, use mean as baseline
                                    success_scores = torch.ones_like(reward_sum) * 0.5
                            
                            new_batch.batch["oppo_success"] = success_scores

                    if not self.config.algorithm.filter_groups.enable:
                        # When filter_groups is disabled, use all trajectories
                        # Align batch size to ensure consistency
                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        batch = new_batch[:traj_bsz]
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

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
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
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
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.OPPO:
                        with marked_timer("oracle_log_prob", timing_raw, "purple"):
                            # OPPO: build two oracle conditions:
                            # - R=1 oracle: prompt augmented with ground truth (correct answer)
                            # - R=0 oracle: prompt augmented with a "wrong" answer (typically the predicted answer from an incorrect traj)
                            oracle_pos_batch = self._build_oracle_prompt_batch(batch)
                            if oracle_pos_batch is None:
                                raise ValueError("Failed to build oracle prompt batch for OPPO.")
                            oracle_pos_log_prob = self.actor_rollout_wg.compute_log_prob(oracle_pos_batch)
                            if "entropys" in oracle_pos_log_prob.batch:
                                oracle_pos_log_prob.batch.pop("entropys")
                            oracle_pos_log_prob.batch["oracle_log_probs"] = oracle_pos_log_prob.batch.pop("old_log_probs")
                            batch = batch.union(oracle_pos_log_prob)

                            # Build R=0 oracle prompts using negative answers if possible.
                            oracle0_pack = self._build_oracle0_prompt_batch_from_pred(batch)
                            if oracle0_pack is not None:
                                oracle0_batch, oracle0_valid = oracle0_pack
                                oracle0_log_prob = self.actor_rollout_wg.compute_log_prob(oracle0_batch)
                                if "entropys" in oracle0_log_prob.batch:
                                    oracle0_log_prob.batch.pop("entropys")
                                oracle0_log_prob.batch["oracle0_log_probs"] = oracle0_log_prob.batch.pop(
                                    "old_log_probs"
                                )
                                batch = batch.union(oracle0_log_prob)
                                batch.batch["oracle0_valid"] = oracle0_valid

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.HSD:
                        with marked_timer("hsd_oracle", timing_raw, "purple"):
                            hsd_cfg = self.config.algorithm.hsd
                            oracle_batch = self._build_oracle_prompt_batch(batch, oracle_cfg=hsd_cfg)
                            if oracle_batch is None:
                                raise ValueError("Failed to build oracle prompt batch for HSD.")

                            batch.batch["hsd_teacher_input_ids"] = oracle_batch.batch["input_ids"]
                            batch.batch["hsd_teacher_attention_mask"] = oracle_batch.batch["attention_mask"]
                            batch.batch["hsd_teacher_position_ids"] = oracle_batch.batch["position_ids"]
                            batch.meta_info["hsd_need_logits"] = True
                            batch.meta_info["hsd_temperature"] = hsd_cfg.get("temperature", 1.0)
                            batch.meta_info["hsd_kl_coef"] = hsd_cfg.get("kl_coef", 0.3)
                            batch.meta_info["hsd_ppo_coef"] = hsd_cfg.get("ppo_coef", 1.0)
                            batch.meta_info["hsd_reward_gate_kl"] = hsd_cfg.get("reward_gate_kl", True)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
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
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values (skip for OPPO/HSD which don't use value network)
                    is_oppo = self.config.algorithm.adv_estimator == AdvantageEstimator.OPPO
                    is_hsd = self.config.algorithm.adv_estimator == AdvantageEstimator.HSD
                    if self.use_critic and not is_oppo and not is_hsd:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout IS weights and mismatch metrics (inherited from RayPPOTrainer)
                    batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                    # IS and mismatch metrics already have mismatch/ prefix
                    metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.OPPO:
                            # OPPO-specific stats (Q_t derived from returns)
                            response_mask = batch.batch["response_mask"].bool()
                            valid_returns = torch.masked_select(batch.batch["returns"], response_mask)
                            if valid_returns.numel() > 0:
                                metrics.update(
                                    {
                                        "oppo/q_mean": torch.mean(valid_returns).detach().item(),
                                        "oppo/q_min": torch.min(valid_returns).detach().item(),
                                        "oppo/q_max": torch.max(valid_returns).detach().item(),
                                    }
                                )
                                # logit(Q) for diagnostics (clamped for stability)
                                q_clamped = torch.clamp(valid_returns, min=1e-6, max=1.0 - 1e-6)
                                logit_q = torch.log(q_clamped) - torch.log1p(-q_clamped)
                                metrics["oppo/logit_q_mean"] = torch.mean(logit_q).detach().item()
                            if "oracle0_valid" in batch.batch:
                                oracle0_valid = batch.batch["oracle0_valid"].float()
                                metrics["oppo/oracle0_valid_ratio"] = torch.mean(oracle0_valid).detach().item()

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.HSD:
                            response_mask = batch.batch["response_mask"].bool()
                            valid_adv = torch.masked_select(batch.batch["advantages"], response_mask)
                            if valid_adv.numel() > 0:
                                metrics.update({
                                    "hsd/reward_adv_mean": valid_adv.mean().detach().item(),
                                    "hsd/reward_adv_std": valid_adv.std().detach().item(),
                                    "hsd/reward_adv_pos_frac": (valid_adv > 0).float().mean().detach().item(),
                                })
                            seq_reward = batch.batch["token_level_rewards"].sum(dim=-1)
                            metrics["hsd/correct_frac"] = (seq_reward > 0).float().mean().detach().item()

                    # update critic (skip for OPPO/HSD which don't use value network)
                    if self.use_critic and not is_oppo and not is_hsd:
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
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

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
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic and not is_oppo and not is_hsd))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
