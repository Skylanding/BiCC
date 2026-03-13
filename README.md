# BICC: Binary Iterative Contrastive Conditioning

BICC is a reinforcement learning framework for LLM post-training that combines **Contrastive GRPO** with **Self-Reflective Conditioning (SRC)**. Built on top of the [verl](https://github.com/volcengine/verl) framework.

## Quick Start

### 1. Prerequisites

- Python >= 3.10
- CUDA >= 12.1
- 8x GPUs (default configuration)

### 2. Install

```bash
git clone https://github.com/Skylanding/BiCC.git
cd BiCC

# Install verl framework (editable mode)
pip install -e .

# Install core dependencies
pip install -r requirements.txt

# Install vLLM (rollout engine)
pip install vllm

# Install Flash Attention
pip install flash-attn --no-build-isolation
```

### 3. Prepare Data

Training data should be in `.parquet` format with at least a `prompt` column. Validation data follows the same format.

### 4. Run Training

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

## Training Script Parameters

The full training command is in `recipe/dapo/run_bicc_dapo.sh`. Below are the key parameters grouped by function.

### BICC-Specific Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `contrastive_grpo.enable` | `True` | Enable Contrastive GRPO advantage computation |
| `contrastive_grpo_src.enable` | `True` | Enable Self-Reflective Conditioning |
| `contrastive_grpo_src.symmetric_conditioning` | `True` | Bidirectional conditioning: both yw\|yl and yl\|yw |
| `contrastive_grpo_src.use_self_reflection` | `True` | Create SRC conditioned input pairs |
| `algorithm.adv_estimator` | `remax` | ReMax advantage estimator (greedy baseline) |
| `algorithm.use_kl_in_reward` | `True` | Include KL penalty in reward |
| `algorithm.kl_penalty` | `0.1` | KL penalty coefficient |

### Actor / Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| `actor.optim.lr` | `1e-6` | Learning rate |
| `actor.optim.lr_warmup_steps` | `10` | Warmup steps |
| `actor.optim.weight_decay` | `0.1` | Weight decay |
| `actor.clip_ratio_low` | `0.2` | DAPO asymmetric clip lower bound |
| `actor.clip_ratio_high` | `0.28` | DAPO asymmetric clip upper bound |
| `actor.grad_clip` | `1.0` | Gradient clipping norm |
| `actor.loss_agg_mode` | `token-mean` | Loss aggregation mode |
| `actor.ppo_mini_batch_size` | `8` | PPO mini-batch size |

### Data / Sequence Lengths

| Parameter | Value | Description |
|-----------|-------|-------------|
| `data.max_prompt_length` | `2048` | Maximum prompt length |
| `data.max_response_length` | `3072` | Maximum response length |
| `data.gen_batch_size` | `16` | Generation batch size |
| `data.train_batch_size` | `16` | Training batch size |
| `rollout.n` | `8` | Number of rollouts per prompt |

### Rollout (vLLM)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rollout.name` | `vllm` | Rollout engine |
| `rollout.temperature` | `0.2` | Sampling temperature |
| `rollout.top_p` | `0.7` | Top-p sampling |
| `rollout.gpu_memory_utilization` | `0.60` | vLLM GPU memory fraction |
| `rollout.tensor_model_parallel_size` | `1` | Tensor parallel size |

### Reward

| Parameter | Value | Description |
|-----------|-------|-------------|
| `reward_model.reward_manager` | `dapo` | Reward manager type |
| `reward_model.overlong_buffer.enable` | `True` | Penalize overlong responses |
| `reward_model.overlong_buffer.len` | `1700` | Overlong threshold (tokens) |

### FSDP / Distributed

| Parameter | Value | Description |
|-----------|-------|-------------|
| `actor.fsdp_config.param_offload` | `True` | Offload parameters to CPU |
| `actor.fsdp_config.optimizer_offload` | `True` | Offload optimizer states to CPU |
| `actor.fsdp_config.fsdp_size` | `8` | FSDP sharding group size |
| `trainer.n_gpus_per_node` | `8` | GPUs per node |
| `trainer.nnodes` | `1` | Number of nodes |

### Logging / Checkpointing

| Parameter | Value | Description |
|-----------|-------|-------------|
| `trainer.logger` | `[console,wandb]` | Logging backends |
| `trainer.test_freq` | `100` | Validation frequency (steps) |
| `trainer.save_freq` | `50` | Checkpoint save frequency |
| `trainer.max_actor_ckpt_to_keep` | `3` | Max actor checkpoints to retain |

## Architecture

```
RayPPOTrainer                  (verl/trainer/ppo/ray_trainer.py)
  └── RayDAPOTrainer           (recipe/dapo/dapo_ray_trainer.py)
      └── RefineDAPOTrainer    (recipe/dapo/refine_dapo_trainer.py)
```

### Core Mechanisms

**Contrastive GRPO**: Partitions rollout responses into positive (r=1) and negative (r=0) groups, then computes advantages as:
- A+ = sqrt(n-/n+) for correct responses
- A- = -sqrt(n+/n-) for incorrect responses

**Self-Reflective Conditioning (SRC)**: Conditions generation on alternative responses:
- Forward: log pi(yw | x, yl) — generate correct after seeing wrong
- Reverse: log pi(yl | x, yw) — generate wrong after seeing correct

**Reward-Confidence Correction (RCC)**: Implemented as `clip_cov` and `kl_cov` policy loss variants in `verl/trainer/ppo/core_algos.py`. Computes Cov(Advantage, log pi) at the token level and clips or penalizes high-confidence tokens to prevent gradient domination.

## Directory Structure

```
BiCC/
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
├── README.md
├── recipe/
│   └── dapo/
│       ├── run_bicc_dapo.sh              # Training entry script
│       ├── main_refine_dapo.py           # Python entry: Ray init, worker setup
│       ├── refine_dapo_trainer.py        # RefineDAPOTrainer (Contrastive GRPO + SRC)
│       ├── dapo_ray_trainer.py           # RayDAPOTrainer (DAPO extensions)
│       └── config/
│           ├── refine.yaml               # BICC Hydra config
│           ├── dapo_trainer.yaml
│           └── dapo_megatron_trainer.yaml
└── verl/
    ├── __init__.py
    ├── protocol.py                       # DataProto
    ├── trainer/
    │   ├── ppo/
    │   │   ├── ray_trainer.py            # RayPPOTrainer base class
    │   │   ├── core_algos.py             # Advantage estimators + policy losses
    │   │   ├── metric_utils.py
    │   │   ├── reward.py
    │   │   └── mismatch_helper.py
    │   └── config/                       # Hydra YAML configs
    ├── utils/                            # Utility libraries
    ├── workers/                          # Distributed workers (Actor, Critic, Rollout, Reward)
    ├── models/                           # Model definitions and patches
    └── single_controller/               # Ray controller
```

## License

Apache 2.0 (inherited from [verl](https://github.com/volcengine/verl)).
