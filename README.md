# 🤖 Legged Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0-silver)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [News & Updates](#news-updates)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Steps](#setup-steps)
  - [Docker Usage (Dockerfile + Bash Scripts)](#docker-usage)
- [Usage](#usage)
  - [Prepare Motion Data](#prepare-motion-data)
  - [Training & Play](#training-and-play)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

<a id="overview"></a>
## 📖 Overview

This repository is an extension for legged robot reinforcement learning based on Isaac Lab, which allows to develop in an isolated environment, outside of the core Isaac Lab repository. The RL algorithm is based on the upstream [RSL-RL library](https://github.com/leggedrobotics/rsl_rl) (`rsl-rl-lib >= 5.0.1`); the AMP algorithm is implemented as an **external module** inside this project (`legged_lab/rsl_rl/amp`), so no forked/patched `rsl_rl` is required.

**Key Features:**

- `DeepMimic` for humanoid robots, including Unitree G1.
- `AMP` Adversarial Motion Priors (AMP) for humanoid robots, including Unitree G1. We suggest retargeting the human motion data by [GMR](https://github.com/YanjieZe/GMR).

<a id="demo"></a>
## Demo

* Adversarial Motion Priors for Unitree G1:

https://github.com/user-attachments/assets/ed84a8a3-f349-44ac-9cfd-2baab2265a25

<a id="news-updates"></a>
## 🔥 News & Updates

- 2026/07/11: AMP now walks on **rough terrain**: swapped the foot-catching height-field tiles for smooth Perlin ports (ported from [InstinctLab](https://github.com/project-instinct/instinctlab/)), and added a third-person follow camera (`--follow_cam`) to the Kit play viewport.
- 2026/07/01: Migrated to **Isaac Lab v3.0.0** and **rsl-rl-lib 5.4.1**. AMP is now an **external algorithm module** (`legged_lab/rsl_rl/amp`) selected via `class_name`, so no forked `rsl_rl` is needed.
- 2026/02/09: Add Dockerfile + bash script workflow, including host path requirement for local `rsl_rl`.
- 2025/12/16: Test in Isaac Lab 2.3.1 and RSL-RL 3.2.0.
- 2025/12/05: Use git lfs to store large files, including motion data and robot models.
- 2025/11/23: Add Symmetry data augmentation in AMP training.
- 2025/11/22: New implementation of AMP.
- 2025/11/19: Add DeepMimic for G1.
- 2025/10/14: Update to support rsl_rl v3.1.1. Only walking in flat terrain is supported now.
- 2025/08/24: Support using more steps observations and motion data in AMP training.
- 2025/08/22: Compatible with Isaac Lab 2.2.0.
- 2025/08/21: Add support for retargeting human motion data by [GMR](https://github.com/YanjieZe/GMR).

<a id="installation"></a>
## ⚙️ Installation

<a id="prerequisites"></a>
### Prerequisites

- **Isaac Lab**: Ensure you have installed Isaac Lab `v3.0.0`. Follow the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- **Git LFS**: Required for downloading large model files.

<a id="setup-steps"></a>
### Setup Steps

1.  **Clone the Repository**
    Clone this repository *outside* your existing `IsaacLab` directory to maintain isolation.

    ```bash
    # Option 1: HTTPS
    git clone https://github.com/zitongbai/legged_lab

    # Option 2: SSH
    git clone git@github.com:zitongbai/legged_lab.git

    cd legged_lab
    ```

2.  **Pull Git LFS Assets**
    Install and initialize `git-lfs` on your machine (one-time), then pull large assets (USD models and motion data) for this repository.

    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Install the Package**
    Use the Python interpreter associated with your Isaac Lab installation.

    ```bash
    python -m pip install -e source/legged_lab
    ```

4.  **Install RSL-RL (Upstream)**
    This project uses the upstream `rsl-rl-lib` (no fork required). Isaac Lab v3.0.0
    ships with a compatible version, but AMP is developed against `5.4.1`:

    ```bash
    python -m pip install "rsl-rl-lib>=5.0.1"
    ```

    The AMP algorithm lives inside this project at `source/legged_lab/legged_lab/rsl_rl/amp`
    and is selected at runtime via the config's `class_name`
    (`legged_lab.rsl_rl.amp.ppo_amp:PPOAMP`) — no patching of `rsl_rl` is needed.

<a id="docker-usage"></a>
### Docker Usage (Dockerfile + Bash Scripts)

> **Note:** The Docker workflow below still targets Isaac Lab 2.3.1 and mounts a
> local forked `rsl_rl`. Since AMP is now an in-project external module and this
> code targets Isaac Lab v3.0.0, the Docker files (`docker/.env.base`,
> `docker/run.sh`) need updating to the v3 base image and to drop the `rsl_rl`
> mount. Until then, prefer the direct (non-Docker) install steps above.

If you use the provided Docker workflow, the container will mount local source code and install packages automatically at startup.

#### Host directory requirement for `rsl_rl`

By default, `docker/.env.base` expects `rsl_rl` to be placed next to `legged_lab`:

```text
.../lab_dev/
├── legged_lab/
└── rsl_rl/
```

If your `rsl_rl` is somewhere else, update `RSL_RL_PATH` in `docker/.env.base`.

By default, Isaac Sim caches, logs, data, and documents use the official Docker directory layout under `~/docker/isaac-sim`.

#### Build image

```bash
bash docker/build.sh
```

#### Start container

```bash
# xhost +
bash docker/run.sh
```

At startup, the container will:
- overwrite `.vscode/settings.json` with the container's built-in VS Code settings
- install mounted `rsl_rl` in editable mode (`/workspace/rsl_rl`)
- install mounted `legged_lab` in editable mode (`/workspace/legged_lab/source/legged_lab`)

#### Enter container

```bash
bash docker/enter.sh
```

Default working directory is `/workspace/legged_lab`.

#### Stop / remove container

```bash
bash docker/stop.sh
```

#### Rebuild image after Dockerfile changes

```bash
bash docker/stop.sh
bash docker/build.sh
bash docker/run.sh
```

<a id="usage"></a>
## 🚀 Usage

<a id="prepare-motion-data"></a>
### 1. Prepare Motion Data

We have already provided some off-the-shelf motion data in the `source/legged_lab/legged_lab/data/MotionData` folder for testing.

If you want to add more motion data, you can do so by following the steps below.

1. Retarget human motion data to the robot model. We recommend using [GMR](https://github.com/YanjieZe/GMR) for retargeting human motion data.
2. Put the retargeted motion data in the `temp/gmr_data` folder.
3. Use a helper script to convert the motion data to the required format:

    ```bash
    python scripts/tools/retarget/dataset_retarget.py \
        --robot g1 \
        --input_dir temp/gmr_data/ \
        --output_dir temp/lab_data/ \
        --config_file scripts/tools/retarget/config/g1_29dof.yaml \
        --loop clamp
    ```
4. Move the converted data from `temp/lab_data` to `source/legged_lab/legged_lab/data/MotionData`, and set the `MotionDataCfg` in the config file, e.g., `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py`.

Please refer to the comments in the script for more details about the arguments, and refer to `scripts/tools/retarget/gmr_to_lab.py` for the data format used in this repository.

<a id="training-and-play"></a>
### 2. Training & Play

#### ⚡ Interactive launcher (optional)

Instead of hand-writing the commands below, you can use the interactive
launcher, which scans the repo for tasks, shows live GPU usage, and walks you
through the common options (GPU, `--viz` backend, `max_iterations`, `run_name`,
checkpoint selection, Hydra overrides, …). It then either prints the assembled
command for you to copy, or launches it directly in a tmux session named after
your `run_name`:

```bash
python -m scripts.launch
```

Navigate the menus with the arrow keys (↑/↓, or `j`/`k`) and press `Enter` to
select. Long lists (e.g. many checkpoints) are paged 10 per page: use ←/→ (or
`h`/`l`) to flip pages and `0`-`9` to jump straight to an entry on the current
page; `q` cancels. The tmux mode reads your Python env and proxy from
`scripts/experiments/env.local.sh`
(see [experiments README](scripts/experiments/README.md)).

The sections below document the underlying `train.py` / `play.py` commands.

#### 🎭 DeepMimic

<details>
<summary>Train</summary>

To train the DeepMimic algorithm, you can run the following command:

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-G1-v0 --headless --max_iterations 50000
```

To train on a non-default GPU, set both `--device` and `agent.device`:

```bash
# replace `x` with the gpu id you want to use
python scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-G1-v0 --headless --max_iterations 50000 \
    --device cuda:x agent.device=cuda:x
```

The `max_iterations` can be adjusted based on your needs. For more details about the arguments, run `python scripts/rsl_rl/train.py -h`.

</details>

<details>
<summary>Play</summary>

You can play the trained model in a headless mode and record the video:

```bash
# replace the checkpoint path with the path to your trained model
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-Deepmimic-G1-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

</details>


#### 🏃 Adversarial Motion Priors (AMP)

<details>
<summary>Train</summary>

The AMP task is split into two configs: **flat** terrain (`LeggedLab-Isaac-AMP-Flat-G1-v0`)
and **rough** generator terrain (`LeggedLab-Isaac-AMP-Rough-G1-v0`, with a height scanner and
a terrain-difficulty curriculum). Pick whichever task id you want to train:

```bash
# flat terrain
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-Flat-G1-v0 --headless --max_iterations 50000

# rough terrain
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-Rough-G1-v0 --headless --max_iterations 50000
```

To train on a non-default GPU, set both `--device` and `agent.device`:

```bash
# replace `x` with the gpu id you want to use
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-Rough-G1-v0 --headless --max_iterations 50000 \
    --device cuda:x agent.device=cuda:x
```

Checkpoints are written to `logs/rsl_rl/g1_amp_flat/...` and `logs/rsl_rl/g1_amp_rough/...`
respectively. For more details about the arguments, run `python scripts/rsl_rl/train.py -h`.

</details>

<details>
<summary>Play</summary>

There are two ways to play a trained model, selected by the `--viz` backend.

**Mode 1 — record a video (`--viz kit`).** The Kit visualizer renders the command-velocity
arrows (desired vs. actual base velocity, drawn above the robot) into the recorded video:

```bash
# flat terrain — replace the checkpoint path with the path to your trained model
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Flat-G1-v0 \
    --num_envs 64 --video --viz kit \
    --checkpoint logs/rsl_rl/g1_amp_flat/run_name/model_xxx.pt

# rough terrain
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Rough-G1-v0 \
    --num_envs 64 --video --viz kit \
    --checkpoint logs/rsl_rl/g1_amp_rough/run_name/model_xxx.pt
```

The video will be saved in the `logs/rsl_rl/experiment_name/run_name/videos/play` directory.

To keep the robot centered in the recording, add `--follow_cam` — the Kit viewport
then chases the followed body (default `torso_link`, env 0) in a third-person view:

```bash
# smooth position-only follow (camera keeps a fixed viewing direction)
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Rough-G1-v0 \
    --viz kit --video --follow_cam \
    --checkpoint logs/rsl_rl/g1_amp_rough/run_name/model_xxx.pt

# follow AND rotate with the robot's heading, damping the per-step yaw jitter
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Rough-G1-v0 \
    --viz kit --video --follow_cam --follow_yaw --follow_smooth 0.9 \
    --checkpoint logs/rsl_rl/g1_amp_rough/run_name/model_xxx.pt
```

Tune the shot with `--follow_env` / `--follow_body` / `--follow_offset`; run
`python scripts/rsl_rl/play.py -h` for the full list. The follow camera drives the
Kit viewport only — the Viser backend (Mode 2) manages its own camera and ignores it.

**Mode 2 — interactive visualization (`--viz viser`).** The Viser backend serves a live 3D
view over HTTP (no recording, no display needed) — open the printed URL in a browser. Do not
pass `--video` or `--headless` in this mode (Viser is a kitless backend and needs neither):

```bash
# flat terrain — replace the checkpoint path with the path to your trained model
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Flat-G1-v0 \
    --num_envs 16 --viz viser \
    --checkpoint logs/rsl_rl/g1_amp_flat/run_name/model_xxx.pt

# rough terrain
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Rough-G1-v0 \
    --num_envs 16 --viz viser \
    --checkpoint logs/rsl_rl/g1_amp_rough/run_name/model_xxx.pt
```

On a remote machine, forward the Viser port to your local browser, e.g.
`ssh -L 8080:localhost:8080 <host>`.

</details>


#### 🎬 Animation (motion replay)

The Animation task (`LeggedLab-Isaac-Animation-G1-v0`) is a **pure motion-data replay** —
there is no policy and no training. Every step the robot is kinematically posed directly
from the motion data (gravity/collision disabled), which is handy for visually inspecting
retargeted reference motion before using it for AMP / DeepMimic. Because there is no policy
to load, it runs with a **zero-action agent** (`scripts/zero_agent.py`) instead of `play.py` —
no `--checkpoint` is needed.

<details>
<summary>Replay / Visualize</summary>

The motion dataset to replay is set by `MotionDataCfg` in
`source/legged_lab/legged_lab/tasks/locomotion/animation/config/g1/g1_anim_env_cfg.py`.

**Mode 1 — Kit viewport (`--viz kit`).** Opens the Isaac Sim viewport; red spheres mark
the reference key-body positions:

```bash
python scripts/zero_agent.py --task LeggedLab-Isaac-Animation-G1-v0 \
    --num_envs 16 --viz kit
```

**Mode 2 — interactive Viser (`--viz viser`).** Serves a live 3D view over HTTP — open the
printed URL in a browser (no display needed). Do not pass `--headless` in this mode:

```bash
python scripts/zero_agent.py --task LeggedLab-Isaac-Animation-G1-v0 \
    --num_envs 16 --viz viser
```

On a remote machine, forward the Viser port to your local browser, e.g.
`ssh -L 8080:localhost:8080 <host>`.

</details>

<a id="roadmap"></a>
## 🗺️ Roadmap

- [ ] Add more legged robots, such as Unitree H1
- [x] Self-contact penalty in AMP
- [x] Asymmetric Actor-Critic in AMP
- [x] Symmetric Reward
- [ ] Sim2sim in mujoco
- [ ] Add support for image observations
- [x] Walk in rough terrain with AMP

<a id="citation"></a>
## 📚 Citation

If you find this repository useful in your research, please consider citing it:

```bibtex
@misc{legged_lab,
  author       = {Zitong Bai},
  title        = {Legged Lab: An Isaac Lab Extension for Legged Robot Reinforcement Learning},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/zitongbai/legged_lab}}
}
```

<a id="acknowledgement"></a>
## 🙏 Acknowledgement

We would like to express our gratitude to the following open-source projects:

- [**Isaac Lab**](https://github.com/isaac-sim/IsaacLab) - The foundation of this project.
- [**RSL-RL**](https://github.com/leggedrobotics/rsl_rl) - Reinforcement learning algorithms for legged robots.
- [**AMP_for_hardware**](https://github.com/Alescontrela/AMP_for_hardware) - Inspiration for AMP implementation.
- [**GMR**](https://github.com/YanjieZe/GMR) - Excellent motion retargeting library.
- [**MimicKit**](https://github.com/xbpeng/MimicKit) - Reference for imitation learning.
- [**InstinctLab**](https://github.com/project-instinct/instinctlab/) - Reference for the Perlin-augmented terrain generation.
