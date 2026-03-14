# 🤖 Legged Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.1-silver)](https://isaac-sim.github.io/IsaacLab/v2.3.1/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
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
  - [Docker Usage (Isaac Lab Image)](#docker-usage)
- [Usage](#usage)
  - [Prepare Motion Data](#prepare-motion-data)
  - [Training & Play](#training-and-play)
- [Roadmap](#roadmap)
- [Acknowledgement](#acknowledgement)

<a id="overview"></a>
## 📖 Overview

This repository is an extension for legged robot reinforcement learning based on Isaac Lab, which allows to develop in an isolated environment, outside of the core Isaac Lab repository. The RL algorithm is based on a [forked RSL-RL library](https://github.com/zitongbai/rsl_rl/tree/feature/amp). 

**Key Features:**

- `DeepMimic` for humanoid robots, including Unitree G1.
- `AMP` Adversarial Motion Priors (AMP) for humanoid robots, including Unitree G1. We suggest retargeting the human motion data by [GMR](https://github.com/YanjieZe/GMR).

<a id="demo"></a>
## Demo

* Adversarial Motion Priors for Unitree G1:

https://github.com/user-attachments/assets/ed84a8a3-f349-44ac-9cfd-2baab2265a25

<a id="news-updates"></a>
## 🔥 News & Updates

- 2026/02/09: Add Docker Compose usage guide (official Isaac Lab image workflow), including host path requirement for local `rsl_rl`.
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

- **Isaac Lab**: Ensure you have installed Isaac Lab `v2.3.1`. Follow the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
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

4.  **Install RSL-RL (Forked Version)**
    We use a customized version of `rsl_rl` to support advanced features like AMP.

    ```bash
    # Clone outside of IsaacLab and legged_lab directories
    git clone -b feature/amp https://github.com/zitongbai/rsl_rl.git
    
    cd rsl_rl
    python -m pip install -e .
    ```

<a id="docker-usage"></a>
### Docker Usage (Isaac Lab Image)

If you use the provided Docker Compose workflow, the container will mount local source code and install packages automatically at startup.

#### Host directory requirement for `rsl_rl`

By default, `docker/docker-compose.yaml` expects `rsl_rl` to be placed next to `legged_lab`:

```text
.../lab_dev/
├── legged_lab/
└── rsl_rl/
```

If your `rsl_rl` is somewhere else, update `RSL_RL_PATH` in `docker/.env.base`.

#### Start container

```bash
docker compose -f docker/docker-compose.yaml up -d
```

At startup, the container will:
- install mounted `rsl_rl` in editable mode (`/workspace/rsl_rl`)
- install mounted `legged_lab` in editable mode (`/workspace/legged_lab/source/legged_lab`)

#### Enter container

```bash
docker compose -f docker/docker-compose.yaml exec legged-lab bash
```

Default working directory is `/workspace/legged_lab`.

#### Stop / remove container

```bash
docker compose -f docker/docker-compose.yaml stop
docker compose -f docker/docker-compose.yaml down
```

#### Recreate container after compose changes

```bash
docker compose -f docker/docker-compose.yaml down
docker compose -f docker/docker-compose.yaml up -d
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

<a id="g1"></a>
### G1

#### 🎭 DeepMimic

<details>
<summary>Train</summary>

To train the DeepMimic algorithm, you can run the following command:

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-G1-v0 --headless --max_iterations 50000
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

To train the AMP algorithm, you can run the following command:

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000
```

If you want to train it in a non-default gpu, you can pass more arguments to the command:

```bash
# replace `x` with the gpu id you want to use
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000 --device cuda:x agent.device=cuda:x
```

For more details about the arguments, run `python scripts/rsl_rl/train.py -h`.

</details>

<details>
<summary>Play</summary>

You can play the trained model in a headless mode and record the video: 

```bash
# replace the checkpoint path with the path to your trained model
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

The video will be saved in the `logs/rsl_rl/experiment_name/run_name/videos/play` directory.

</details>

<a id="go2"></a>
### Go2

#### Local Navigation

<details>
<summary>Train</summary>

To train on different terrains, you can run the following command:

```bash
python scripts/rsl_rl/train.py --task=Lab-Position-Rough-Unitree-Go2-v0 --headless --device cuda:x agent.device=cuda:x
```

If you want to train it using symmetry data augmentation, you can run the following command:

```bash
python scripts/rsl_rl/train.py --task=Lab-Position-Rough-Unitree-Go2-v0 --headless --agent=rsl_rl_with_symmetry_cfg_entry_point --run_name=with_symmetry
```

For more types of terrains, you can replace "Rough" with "Pit" or "Gap" in the commands above.

</details>

<details>
<summary>Play</summary>

You can play the trained model in a headless mode and record the video: 

```bash
# replace the checkpoint path with the path to your trained model
python scripts/rsl_rl/play.py --task=Lab-Position-Rough-Unitree-Go2-Play-v0 --headless --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

</details>

#### 🏃 Local Navigation with Adversarial Motion Priors (AMP)

<details>
<summary>Train</summary>

To train the AMP algorithm, you can run the following command:

```bash
python scripts/rsl_rl/train.py --task Lab-Position-Rough-AMP-Go2-v0 --headless --max_iterations 50000
```

For more types of terrains, you can replace "Rough" with "Pit" or "Gap" in the commands above.

</details>

<details>
<summary>Play</summary>

You can play the trained model in a headless mode and record the video: 

```bash
# replace the checkpoint path with the path to your trained model
python scripts/rsl_rl/play.py --task Lab-Position-Rough-AMP-Go2-Play-v0 --headless --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

The video will be saved in the `logs/rsl_rl/experiment_name/run_name/videos/play` directory.

</details>

<a id="roadmap"></a>
## 🗺️ Roadmap

- [ ] Add more legged robots, such as Unitree H1
- [x] Self-contact penalty in AMP
- [x] Asymmetric Actor-Critic in AMP
- [x] Symmetric Reward
- [ ] Sim2sim in mujoco
- [ ] Add support for image observations
- [ ] Walk in rough terrain with AMP

<a id="acknowledgement"></a>
## 🙏 Acknowledgement

We would like to express our gratitude to the following open-source projects:

- [**Isaac Lab**](https://github.com/isaac-sim/IsaacLab) - The foundation of this project.
- [**RSL-RL**](https://github.com/leggedrobotics/rsl_rl) - Reinforcement learning algorithms for legged robots.
- [**AMP_for_hardware**](https://github.com/Alescontrela/AMP_for_hardware) - Inspiration for AMP implementation.
- [**GMR**](https://github.com/YanjieZe/GMR) - Excellent motion retargeting library.
- [**MimicKit**](https://github.com/xbpeng/MimicKit) - Reference for imitation learning.
