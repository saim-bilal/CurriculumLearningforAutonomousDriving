# CurriculumAutoNav

> **Curriculum vs. Non-Curriculum Reinforcement Learning for Autonomous Driving**

A reproducible research framework comparing staged curriculum learning against direct composite-map training for autonomous driving agents. Experiments are run across two simulators—**Highway-Env** and **MetaDrive**—using three RL algorithms: PPO, DQN, and a custom hand-written SimpleDQN.

---

## Table of Contents

1. [What the Project Does](#what-the-project-does)
2. [Key Results](#key-results)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
6. [Usage](#usage)
7. [Experiment Design](#experiment-design)
8. [Conclusion](#conclusion)
9. [Contributions](#contributions)
10. [Where to Get Help](#where-to-get-help)
11. [Maintainers & Contributing](#maintainers--contributing)

---

## What the Project Does

CurriculumAutoNav investigates whether **curriculum learning**—training agents on progressively harder driving scenarios—produces better-performing and safer autonomous driving policies compared to training directly on the hardest scenario (the non-curriculum baseline).

### Simulators

| Simulator                                                      | Purpose                                                        |
| -------------------------------------------------------------- | -------------------------------------------------------------- |
| [Highway-Env](https://github.com/Farama-Foundation/HighwayEnv) | Highway, merge, intersection, and roundabout maps              |
| [MetaDrive](https://github.com/metadriverse/metadrive)         | Procedurally generated roads with configurable traffic density |

### Algorithms

| Algorithm | Source            | Notes                                         |
| --------- | ----------------- | --------------------------------------------- |
| PPO       | Stable-Baselines3 | On-policy, actor-critic                       |
| DQN       | Stable-Baselines3 | Off-policy, experience replay                 |
| SimpleDQN | Custom (PyTorch)  | Handcrafted SB3-compatible agent for ablation |

---

## Key Results

Highway-Env experiments (seed 0, 100k timesteps):

| Algorithm | Approach       | Mean Reward | Success Rate | Collision Rate |
| --------- | -------------- | ----------- | ------------ | -------------- |
| PPO       | **Curriculum** | **99.9**    | **85%**      | 15%            |
| PPO       | Non-Curriculum | 68.9        | 70%          | 30%            |
| DQN       | Curriculum     | 66.7        | 50%          | 50%            |
| DQN       | Non-Curriculum | 60.0        | 50%          | 50%            |
| SimpleDQN | Curriculum     | 68.5        | 45%          | 55%            |
| SimpleDQN | Non-Curriculum | 64.8        | 57%          | 43%            |

Curriculum PPO achieves a **+45% reward gain** and **+15 pp success-rate improvement** over the non-curriculum baseline on Highway-Env.

---

## Features

* Adaptive curriculum progression (5k-step chunks; success thresholds)
* Multi-block scenario stitching (highway → merge → intersection → roundabout)
* Stage-dependent reward shaping (collision/off-road/traffic penalties)
* Research-grade metrics (success, collision, completion, avg. speed, sample efficiency)
* Statistical analysis (paired t-tests, Cohen's d, 95% CI)
* TensorBoard integration and auto-generated comparison artifacts
* Held-out generalization tests on unseen maps/traffic densities

---

## Project Structure

```
code/
├── HighwayEnv/
│   ├── highwayenv_v3.ipynb        # End-to-end Highway-Env experiment notebook
│   └── experiments/               # Saved results for PPO / DQN / SimpleDQN
│       ├── comparisons/           # Auto-generated CSV + plots
│       ├── DQN/
│       ├── PPO/
│       └── SimpleDQN/
└── MetaDrive/
    ├── curriculum/
    │   └── curriculum.ipynb       # MetaDrive curriculum pipeline (PPO / DQN / SAC)
    └── non_curriculum/
        └── non_curriculum.ipynb   # MetaDrive non-curriculum baseline
```

> **Paper:** `submissions/Final_report_G17` (final project report)

---

## Getting Started

### Prerequisites

* Python 3.10 or 3.12 (MetaDrive prefers 3.12)
* Jupyter / VS Code with Jupyter extension

### Installation — Highway-Env

Run the first cell of `HighwayEnv/highwayenv_v3.ipynb`, or install manually:

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium highway-env stable-baselines3==2.1.0 \
            matplotlib pandas seaborn scipy numpy tensorboard
```

### Installation — MetaDrive

Run the first cell of `MetaDrive/curriculum/curriculum.ipynb`, or:

```bash
# Google Colab / default
pip install "stable-baselines3[extra]" "metadrive-simulator-py3-12" \
            tensorboard opencv-python

# Kaggle (NumPy < 2 required)
pip install "numpy<2.0" "protobuf==3.20.3" "metadrive-simulator" \
            "stable-baselines3[extra]" tensorboard opencv-python
```

> **Windows note:** Set `N_ENVS = 1` in configuration cells to avoid multiprocessing issues in Jupyter.

---

## Usage

### Running Highway-Env experiments

Open `HighwayEnv/highwayenv_v3.ipynb` and execute cells in order (install → set timesteps/hyperparams → run). Use:

```python
# Run all algorithms, single seed
main(seed=0)
```

Results are written to `HighwayEnv/experiments/<ALGO>/<approach>/seed_<N>/run_<timestamp>/`.

### Running MetaDrive experiments

Open the relevant notebook and run all cells:

| Goal                    | Notebook                                        |
| ----------------------- | ----------------------------------------------- |
| Curriculum (C0 → C3)    | `MetaDrive/curriculum/curriculum.ipynb`         |
| Non-curriculum baseline | `MetaDrive/non_curriculum/non_curriculum.ipynb` |

### Viewing TensorBoard logs

```bash
tensorboard --logdir MetaDrive/curriculum/experiments_PPO/tensorboard
```

---

## Experiment Design

### Highway-Env Curriculum Stages

| Stage | Maps                         | Traffic density |
| ----- | ---------------------------- | --------------- |
| 1     | `highway-v0`                 | 20 %            |
| 2     | `highway-v0` + `merge-v0`    | 25 – 30 %       |
| 3     | + `intersection-v0`          | 30 – 35 %       |
| 4     | + `roundabout-v0` (shuffled) | 35 – 45 %       |

The non-curriculum baseline trains directly on Stage 4 for the same total timestep budget.

### MetaDrive Curriculum Stages

| Stage | Map                 | Traffic      |
| ----- | ------------------- | ------------ |
| C0    | Straight road (`S`) | None         |
| C1    | Roundabout (`O`)    | None         |
| C2    | 10-block PG map     | Light (5 %)  |
| C3    | 20-block PG map     | Dense (30 %) |

Total curriculum budget: **650,000 timesteps** (100k + 150k + 200k + 200k).

### Key Hyperparameters

| Param         | PPO                       | DQN        | SimpleDQN  |
| ------------- | ------------------------- | ---------- | ---------- |
| Learning rate | 5e-4                      | 1e-4       | 5e-4       |
| Network arch  | [256, 256] (separate π/V) | [256, 256] | [256, 256] |
| Buffer size   | —                         | 100k       | 50k        |
| Discount γ    | 0.99                      | 0.99       | 0.99       |

---

## Conclusion

* **Main finding:** Curriculum learning can accelerate training and improve final performance for some RL algorithms (notably PPO) in autonomous driving benchmarks, but its benefits are context‑ and algorithm‑dependent.
* **Caveats:** Curriculum stages can induce overfitting to intermediate tasks and catastrophic forgetting; value‑based agents are prone to reward hacking in sparse or poorly shaped reward regimes; handcrafted agents with aggressive hyperparameters may destabilize under stage transitions.
* **Recommendations & future work:** Design algorithm‑specific curricula, incorporate stage‑aware replay or regularization to mitigate forgetting, apply dynamic reward shaping (penalize passivity in later stages), and explore hybrid architectures that combine value‑based assertiveness with policy‑gradient stability.

---

## Contributions

* **Raahim A. Samad Poonawala** — Conceptualized and implemented the MetaDrive experiments and curriculum design; performed literature review on RL training mechanisms and reward structures; compiled and analyzed MetaDrive results for the report.
* **Muhammad Bin Tariq** — Led the literature review and authored the paper’s initial sections (abstract, introduction, problem formulation); conducted preliminary simulations and set up HighwayEnv experiments.
* **Saim Bilal (me)** — Proposed the curriculum‑learning idea; executed the full HighwayEnv training regimen (curriculum and non‑curriculum); compiled and authored the HighwayEnv results and discussion.

---

## Where to Get Help

* **Highway-Env docs**: [https://highway-env.farama.org](https://highway-env.farama.org)
* **MetaDrive docs**: [https://metadrive-simulator.readthedocs.io](https://metadrive-simulator.readthedocs.io)
* **Stable-Baselines3 docs**: [https://stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)
* **Gymnasium docs**: [https://gymnasium.farama.org](https://gymnasium.farama.org)

---

## Maintainers & Contributing

This project was developed as a final research submission for the course CS 6314: Dynamic Programming and Reinforcement Learning at LUMS.

### Contribution guidelines

1. Fork the repository and create a feature branch.
2. Keep notebooks self-contained—all imports and `pip install` calls at the top.
3. Follow the existing directory convention: `experiments/<ALGO>/<approach>/seed_<N>/run_<timestamp>/`.
4. Add new algorithms by implementing the SB3-compatible `learn(total_timesteps, ...)` / `predict(obs, ...)` / `set_env(env)` / `save(path)` interface (see `SimpleDQNAgent` for a reference implementation).
5. Open a pull request with a brief description of what changed and why.

