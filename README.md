# BICC - Binary Iterative Contrastive Conditioning

BICC 训练框架的核心代码提取，包含完整的依赖链。

源自 [verl (Volcano Engine Reinforcement Learning for LLMs)](https://github.com/verl-project/verl)。

---

## 环境配置

### 1. 基础环境要求

- Python >= 3.10
- CUDA >= 12.1（推荐 12.4）
- 8 x GPU（脚本默认 8 卡，可调整）

### 2. 安装 verl 框架

推荐从 GitHub 源码安装（包含最新功能和 BICC 所需的所有依赖）：

```bash
# 克隆 verl 仓库
git clone https://github.com/verl-project/verl.git
cd verl
git submodule update --init --recursive recipe

# 安装 verl（开发模式）
pip install -e .
```

### 3. 安装核心依赖

```bash
# 核心依赖（requirements.txt）
pip install accelerate codetiming datasets dill hydra-core liger-kernel \
  "numpy<2.0.0" pandas peft "pyarrow>=19.0.0" pybind11 pylatexenc \
  "ray[default]" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
  transformers wandb "packaging>=20.0" uvicorn fastapi \
  latex2sympy2_extended math_verify tensorboard

# vLLM（rollout 引擎，推荐 >= 0.8.2，避免 0.7.x）
pip install vllm

# Flash Attention（CUDA 加速）
pip install flash-attn --no-build-isolation
```

### 4. 可选：SGLang 后端

```bash
# 如需使用 SGLang 替代 vLLM 做 rollout
pip install sglang
```

### 5. 验证安装

```bash
python -c "import verl; print(f'verl version: {verl.__version__}')"
python -c "import ray; ray.init(); print('Ray OK'); ray.shutdown()"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

### 6. 运行 BICC 训练

```bash
# 编辑路径配置
vim recipe/dapo/run_bicc_dapo.sh
# 修改以下路径：
#   MODEL_PATH   -> 基座模型路径（如 Qwen3-4B）
#   TRAIN_FILE   -> 训练数据 parquet
#   TEST_FILE    -> 验证数据 parquet
#   CKPTS_DIR    -> checkpoint 输出目录

# 启动训练（在 verl 根目录下执行）
cd /home/ubuntu/verl
bash recipe/dapo/run_bicc_dapo.sh
```

---

## 目录结构（219 个文件，1.7MB）

```
BICC/
├── README.md
│
├── recipe/dapo/                          # ══ 模块1: BICC Trainer ══
│   ├── run_bicc_dapo.sh                  # 启动脚本 (入口)
│   ├── main_refine_dapo.py               # Python 入口: Ray 初始化, worker 创建, 启动训练
│   ├── refine_dapo_trainer.py            # ★ RefineDAPOTrainer (2100+ 行)
│   │                                     #   - fit() 主循环
│   │                                     #   - Contrastive GRPO: A+ = √(n-/n+)
│   │                                     #   - SRC: Self-Reflective Conditioning 配对
│   │                                     #   - format_grpo_as_contrastive() [DPO-style σ(β·Δ)]
│   │                                     #   - create_self_reflective_pairs()
│   │                                     #   - format_grpo_as_contrastive_with_src()
│   │                                     #   - ContrastiveGRPOConfig / ContrastiveGRPOSRCConfig
│   ├── dapo_ray_trainer.py               # RayDAPOTrainer (OPPO oracle batch)
│   ├── main_dapo.py                      # 标准 DAPO 入口 (对照参考)
│   └── config/
│       ├── refine.yaml                   # BICC Hydra 配置
│       ├── dapo_trainer.yaml
│       └── dapo_megatron_trainer.yaml
│
└── verl/                                 # ══ 模块2: verl 框架核心 ══
    ├── __init__.py
    ├── protocol.py                       # DataProto 数据协议
    ├── base_config.py                    # 基础配置
    │
    ├── trainer/
    │   ├── ppo/
    │   │   ├── ray_trainer.py            # ★ RayPPOTrainer 基类
    │   │   ├── core_algos.py             # ★ 核心算法:
    │   │   │                             #   Advantage: GRPO, ReMax, OPPO, HSD, RLOO, R++...
    │   │   │                             #   Policy Loss: clip, clip_cov (RCC), kl_cov (RCC)
    │   │   ├── metric_utils.py           # 训练指标
    │   │   ├── reward.py                 # load_reward_manager
    │   │   ├── mismatch_helper.py        # Rollout IS weights
    │   │   └── utils.py                  # Role, WorkerType
    │   ├── config/                       # Hydra 配置体系 (30+ yaml)
    │   └── main_ppo.py
    │
    ├── utils/                            # 工具库
    │   ├── torch_functional.py           # masked_mean, masked_whiten
    │   ├── groupwise.py                  # group_mean_std, as_torch_index
    │   ├── profiler/                     # marked_timer 性能 profiler
    │   ├── debug/                        # 调试工具
    │   ├── checkpoint/                   # Checkpoint 管理
    │   ├── dataset/                      # RL 数据集
    │   ├── metric/                       # reduce_metrics
    │   ├── reward_score/                 # 各类 reward 评分 (math, code...)
    │   ├── tracking.py                   # Wandb/console logger
    │   ├── distributed.py                # 分布式工具
    │   ├── fsdp_utils.py                 # FSDP 工具
    │   ├── rollout_skip.py
    │   ├── tokenizer.py                  # hf_tokenizer, hf_processor
    │   └── ...
    │
    ├── workers/                          # 分布式 Worker
    │   ├── fsdp_workers.py               # FSDP ActorRolloutRefWorker
    │   ├── megatron_workers.py           # Megatron workers
    │   ├── actor/                        # Actor worker
    │   ├── critic/                       # Critic worker
    │   ├── rollout/                      # Rollout worker
    │   ├── engine/                       # Training engine
    │   ├── reward_manager/               # DAPORewardManager, registry
    │   ├── reward_model/                 # Reward model worker
    │   ├── sharding_manager/             # FSDP sharding
    │   └── config/                       # ActorConfig, etc.
    │
    ├── models/                           # 模型定义
    │   ├── registry.py
    │   ├── transformers/                 # HF monkey patches
    │   ├── mcore/                        # Megatron-Core bridge
    │   └── llama/megatron/               # LLaMA Megatron layers
    │
    ├── single_controller/                # Ray 控制器
    │   ├── base/                         # 基础装饰器
    │   └── ray/                          # Ray worker group
    │
    └── experimental/                     # 实验性功能
        ├── dataset/                      # CurriculumSampler
        └── agent_loop/                   # Agent loop
```

## 继承链

```
RayPPOTrainer           (verl/trainer/ppo/ray_trainer.py)
  └── RayDAPOTrainer    (recipe/dapo/dapo_ray_trainer.py)
      └── RefineDAPOTrainer  (recipe/dapo/refine_dapo_trainer.py)  ← BICC 实际使用
```

## BICC 的核心机制

### 1. Contrastive GRPO + SRC (refine_dapo_trainer.py)

- **Contrastive Advantage**: A⁺ = √(n⁻/n⁺)，A⁻ = -√(n⁺/n⁻)
- **Self-Reflective Conditioning**: π(yw | x, yl) 和 π(yl | x, yw)
- **DPO-style Sigmoid** (format_grpo_as_contrastive): σ(β·Δ_pair) 置信度修正

### 2. Reward-Confidence Correction / RCC (core_algos.py)

论文公式:
```
δ(x, y) = log π_θ(y|x) - log π_ref(y|x)
b* ≈ E[R] + 2·Cov(R, δ)
```

代码中对应的实现 (`clip_cov` / `kl_cov`):
```
Cov(Advantage, log π) → 识别高置信度-高优势 token → clipping / KL penalty
```
- `clip_cov`: 对高 Cov(A, logπ) token 直接归零 loss
- `kl_cov`: 对高 Cov(A, logπ) token 施加 KL penalty

### 3. run_bicc_dapo.sh 关键配置

| 参数 | 值 | 含义 |
|------|-----|------|
| `adv_estimator` | remax | ReMax advantage (greedy baseline) |
| `use_kl_in_reward` | True | Reward 中包含 KL penalty |
| `contrastive_grpo.enable` | True | 启用 Contrastive GRPO |
| `contrastive_grpo_src.enable` | True | 启用 SRC 条件化 |
| `symmetric_conditioning` | True | 对称交叉条件 |
| `rollout.n` | 8 | 每 prompt 8 次 rollout |
| `clip_ratio_low/high` | 0.2/0.28 | DAPO 非对称 clipping |
| `overlong_buffer.len` | 1700 | 超长回复惩罚阈值 |

---

提取自: `/home/ubuntu/verl/`
提取时间: 2026-03-13
