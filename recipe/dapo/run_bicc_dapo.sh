#!/usr/bin/env bash
set -xeuo pipefail

# BICC Training Script
# Usage: ./run_bicc_dapo.sh
# This script demonstrates BICC training

echo "Starting BICC training"

# Set environment variables (using 8 GPUs)
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export VLLM_USE_V1="1"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Model and data paths
export MODEL_PATH="/path/to/your/base_model"
export TRAIN_FILE="/path/to/train.parquet"
export TEST_FILE="/path/to/val.parquet"

# Set checkpoint directory and experiment names
export CKPTS_DIR="/path/to/checkpoints/BICC-Qwen3-4B-8GPU"
export EXPERIMENT_NAME="BICC-Qwen3-4B"
export PROJECT_NAME="BICC"
echo "📝 Using RefineDAPOTrainer with BICC"

# Create checkpoint directory
mkdir -p "${CKPTS_DIR}"

# Length configuration
max_prompt_length=$((1024 * 2))  # 2048
max_response_length=$((1024 * 3))  # 3072
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 6 / 5))  # 1.2x (6144)
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3 / 2))  # 1.5x (7680)

# Training parameters
cd /home/ubuntu/verl

# Base training command
BASE_CMD="python3 -m recipe.dapo.main_refine_dapo"

BASE_CMD="${BASE_CMD} \
    data.train_files=\"${TRAIN_FILE}\" \
    data.val_files=\"${TEST_FILE}\" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=16 \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.adv_estimator=remax \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=0.1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=0 \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path=\"${MODEL_PATH}\" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    +actor_rollout_ref.model.override_config.attn_implementation=\"eager\" \
    +actor_rollout_ref.model.override_config.torch_dtype=\"bfloat16\" \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=\"token-mean\" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.temperature=0.2 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.2 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=1700 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    contrastive_grpo.enable=True \
    contrastive_grpo.epsilon=1e-8 \
    contrastive_grpo.handle_uniform_rewards=True \
    contrastive_grpo.log_advantage_stats=True \
    contrastive_grpo_src.enable=True \
    contrastive_grpo_src.use_self_reflection=True \
    contrastive_grpo_src.symmetric_conditioning=True \
    trainer.logger='[console,wandb]' \
    trainer.project_name=\"${PROJECT_NAME}\" \
    trainer.experiment_name=\"${EXPERIMENT_NAME}\" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=100 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=\"${CKPTS_DIR}\" \
    trainer.resume_mode=disable"

# Execute the command
eval "${BASE_CMD}"

echo "BICC training completed!"
echo "Checkpoint directory: ${CKPTS_DIR}"

