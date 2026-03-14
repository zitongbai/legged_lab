# Configurable AMP Terminal Preview Groups Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make AMP terminal observation preview groups configurable from the environment config so the env exports only the groups needed by the algorithm.

**Architecture:** Add a `terminal_obs_groups` field to the AMP env config, extend the preview observation manager with selective preview support, and update `ManagerBasedAmpEnv.step()` to preview and export only configured groups. Keep `rsl_rl` unchanged because it already consumes the relevant env observation groups.

**Tech Stack:** Python, Isaac Lab manager-based environments, pytest via `bash .codex/run-in-env.sh`

---

### Task 1: Add a failing test for selective terminal preview export

**Files:**
- Modify: `source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py`

**Step 1: Write the failing test**

Add a test that configures `terminal_obs_groups=("disc",)` and asserts:
- `preview_group("disc")` is called
- `preview_group("policy")` is not called
- `extras["terminal_obs"]` contains only `disc`

**Step 2: Run test to verify it fails**

Run: `bash .codex/run-in-env.sh pytest source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py -q`

Expected: FAIL because the env currently previews all groups.

### Task 2: Implement configurable terminal preview groups

**Files:**
- Modify: `source/legged_lab/legged_lab/envs/manager_based_amp_env_cfg.py`
- Modify: `source/legged_lab/legged_lab/managers/preview_observation_manager.py`
- Modify: `source/legged_lab/legged_lab/envs/manager_based_amp_env.py`

**Step 1: Add config**

Add `terminal_obs_groups: tuple[str, ...] = ("disc",)` to the AMP env config.

**Step 2: Add selective preview support**

Allow the preview manager to preview either all groups or a provided iterable of group names.

**Step 3: Update AMP env step**

Preview only configured groups and export only those groups in `extras["terminal_obs"]`.

**Step 4: Run test to verify it passes**

Run: `bash .codex/run-in-env.sh pytest source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py -q`

Expected: PASS

### Task 3: Verify no regressions in existing terminal preview behavior

**Files:**
- Test: `source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py`

**Step 1: Run targeted test file**

Run: `bash .codex/run-in-env.sh pytest source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py -q`

Expected: PASS
