<p align="center">
  <h1 align="center">
  BICC: Binary Iterative Contrastive Conditioning<br>
  <sub><sub></sub></sub>
  </h1>
  <p align="center">
    <!-- Authors -->
    <!-- <strong>Author Name</strong><sup>1</sup> -->
    <br>
    <!-- Affiliations -->
    <!-- <sup>1</sup>Institution -->
    <br>
    <!-- <a href=''><img src='https://img.shields.io/badge/ArXiv-XXXX.XXXXX-red'></a>&nbsp; -->
    <a href='https://github.com/Skylanding/BiCC'><img src='https://img.shields.io/badge/GitHub-Code-black?logo=github'></a>&nbsp;
    <br>
    <!-- <img src="figure/overview.png"> -->
  </p>
  <br>
</p>

## Abstract

<!-- TODO: Add paper abstract here -->

## Installation

```bash
git clone https://github.com/Skylanding/BiCC.git
cd BiCC

pip install -e .
pip install -r requirements.txt
pip install vllm
pip install flash-attn --no-build-isolation
```

**Requirements:** Python >= 3.10, CUDA >= 12.1, 8x GPUs.

## Training

Edit paths in `recipe/dapo/run_bicc_dapo.sh`:

```bash
export MODEL_PATH="/path/to/your/base_model"   # e.g., Qwen3-4B
export TRAIN_FILE="/path/to/train.parquet"
export TEST_FILE="/path/to/val.parquet"
export CKPTS_DIR="/path/to/checkpoints"
```

Then launch:

```bash
bash recipe/dapo/run_bicc_dapo.sh
```

### Key Parameters

The full command is in `recipe/dapo/run_bicc_dapo.sh`. Key parameters:

```bash
python3 -m recipe.dapo.main_refine_dapo \
    # ── BICC Core ──
    contrastive_grpo.enable=True \
    contrastive_grpo_src.enable=True \
    contrastive_grpo_src.use_self_reflection=True \
    contrastive_grpo_src.symmetric_conditioning=True \
    algorithm.adv_estimator=remax \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=0.1 \
    # ── Data ──
    data.max_prompt_length=2048 \
    data.max_response_length=3072 \
    data.gen_batch_size=16 \
    data.train_batch_size=16 \
    actor_rollout_ref.rollout.n=8 \
    # ── Actor / Optimization ──
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    # ── Rollout (vLLM) ──
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.2 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    # ── Reward ──
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=1700 \
    # ── FSDP ──
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    # ── Trainer ──
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=100 \
    trainer.total_epochs=1
```

## Architecture

```
RayPPOTrainer                  verl/trainer/ppo/ray_trainer.py
  └── RayDAPOTrainer           recipe/dapo/dapo_ray_trainer.py
      └── RefineDAPOTrainer    recipe/dapo/refine_dapo_trainer.py
```

**Contrastive GRPO** — Partitions rollout responses into positive (r=1) and negative (r=0) groups:
  A⁺ = √(n⁻/n⁺),  A⁻ = −√(n⁺/n⁻)

**Self-Reflective Conditioning (SRC)** — Conditions generation on alternative responses:
  Forward: log π(y_w | x, y_l),  Reverse: log π(y_l | x, y_w)

**Reward-Confidence Correction (RCC)** — Token-level covariance clipping (`clip_cov` / `kl_cov`) in `core_algos.py` to prevent high-confidence samples from dominating gradients.

## Project Structure

```
BiCC/
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
├── recipe/
│   └── dapo/
│       ├── run_bicc_dapo.sh           # Training entry script
│       ├── main_refine_dapo.py        # Python entry point
│       ├── refine_dapo_trainer.py     # RefineDAPOTrainer
│       ├── dapo_ray_trainer.py        # RayDAPOTrainer
│       └── config/
│           └── refine.yaml            # Hydra config
└── verl/                              # verl framework (core dependencies)
    ├── trainer/ppo/                   # PPO trainer, core_algos, metrics
    ├── workers/                       # Actor, Critic, Rollout, Reward workers
    ├── utils/                         # Utilities
    ├── models/                        # Model definitions
    └── single_controller/             # Ray controller
```

## Citation

```bibtex
<!-- TODO -->
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
