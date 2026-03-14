# AMP Terminal Disc Preview Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a local observation-manager preview capability and wire AMP to use terminal discriminator observations without modifying the Isaac Lab source tree.

**Architecture:** The implementation adds a `legged_lab` observation-manager subclass with a non-mutating preview API, swaps that manager into `ManagerBasedAmpEnv`, exports terminal discriminator observations through `extras`, and updates local `rsl_rl` AMP processing to consume them for terminated environments only.

**Tech Stack:** Python, Isaac Lab manager-based environments, local `rsl_rl` AMP implementation, pytest via `bash .codex/run-in-env.sh`

---

### Task 1: Add observation-manager preview tests

**Files:**
- Create: `source/legged_lab/test/test_preview_observation_manager.py`
- Reference: `/home/xiaobaige/isaaclab/v2.3.1/IsaacLab/source/isaaclab/test/managers/test_observation_manager.py`

**Step 1: Write the failing test**

Add Isaac-Lab-style pytest coverage for:
- preview of a history-based concatenated group includes the current observation as the newest frame
- preview does not mutate the internal history buffer
- preview of a non-history term matches the normal processed observation value

**Step 2: Run test to verify it fails**

Run: `bash .codex/run-in-env.sh pytest source/legged_lab/test/test_preview_observation_manager.py -v`

Expected: FAIL because the preview-capable observation manager does not exist yet.

**Step 3: Write minimal implementation**

Create a local observation-manager subclass in `source/legged_lab/legged_lab/managers/` and export it from the package.

**Step 4: Run test to verify it passes**

Run: `bash .codex/run-in-env.sh pytest source/legged_lab/test/test_preview_observation_manager.py -v`

Expected: PASS

### Task 2: Swap the manager into AMP env and export terminal disc observations

**Files:**
- Modify: `source/legged_lab/legged_lab/envs/manager_based_amp_env.py`
- Modify: `source/legged_lab/legged_lab/managers/__init__.py`
- Create or modify: `source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py`
- Reference: `/home/xiaobaige/isaaclab/v2.3.1/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py`
- Reference: `/home/xiaobaige/isaaclab/v2.3.1/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`

**Step 1: Write the failing test**

Add a focused environment test that verifies:
- `ManagerBasedAmpEnv` instantiates the local preview-capable observation manager
- `extras["terminal_disc_obs"]` is produced for terminated environments
- returned `obs["disc"]` still has the normal post-reset semantics

**Step 2: Run test to verify it fails**

Run: `bash .codex/run-in-env.sh pytest source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py -v`

Expected: FAIL because `ManagerBasedAmpEnv` does not yet export terminal discriminator observations.

**Step 3: Write minimal implementation**

Override `ManagerBasedAmpEnv.load_managers()` to preserve Isaac Lab manager construction order while substituting the local observation manager. In `step()`, compute terminal `disc` previews before reset and store them in `self.extras`.

**Step 4: Run test to verify it passes**

Run: `bash .codex/run-in-env.sh pytest source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py -v`

Expected: PASS

### Task 3: Make PPOAMP consume terminal discriminator observations

**Files:**
- Modify: `/home/xiaobaige/projects/lab_dev/rsl_rl/rsl_rl/algorithms/ppo_amp.py`
- Create: `/home/xiaobaige/projects/lab_dev/rsl_rl/tests/test_ppo_amp_terminal_disc_obs.py`

**Step 1: Write the failing test**

Add a focused algorithm test that verifies:
- when `extras` contains `terminal_disc_obs`, `PPOAMP.process_env_step()` uses it for `dones == True`
- non-terminated environments keep the default discriminator observation
- the replaced discriminator observation is what drives both style reward computation and replay-buffer insertion

**Step 2: Run test to verify it fails**

Run: `bash .codex/run-in-env.sh pytest /home/xiaobaige/projects/lab_dev/rsl_rl/tests/test_ppo_amp_terminal_disc_obs.py -v`

Expected: FAIL because `PPOAMP` currently ignores `terminal_disc_obs`.

**Step 3: Write minimal implementation**

Update `PPOAMP.process_env_step()` to replace done-env discriminator observations from `extras["terminal_disc_obs"]` before computing style rewards and appending to `disc_obs_buffer`.

**Step 4: Run test to verify it passes**

Run: `bash .codex/run-in-env.sh pytest /home/xiaobaige/projects/lab_dev/rsl_rl/tests/test_ppo_amp_terminal_disc_obs.py -v`

Expected: PASS

### Task 4: Run targeted regression verification

**Files:**
- Verify only

**Step 1: Run the new targeted tests together**

Run: `bash .codex/run-in-env.sh pytest source/legged_lab/test/test_preview_observation_manager.py source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py /home/xiaobaige/projects/lab_dev/rsl_rl/tests/test_ppo_amp_terminal_disc_obs.py -v`

Expected: PASS

**Step 2: Run formatting and lint hooks for touched files**

Run: `bash .codex/run-in-env.sh pre-commit run --files source/legged_lab/legged_lab/managers/__init__.py source/legged_lab/legged_lab/envs/manager_based_amp_env.py source/legged_lab/test/test_preview_observation_manager.py source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py`

Expected: PASS

**Step 3: Commit**

Commit after showing the message to the user and getting confirmation, per repository policy.
