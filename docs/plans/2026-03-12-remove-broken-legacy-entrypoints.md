# Remove Broken Legacy Entrypoints Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove broken legacy scripts and test scripts that still reference deleted AMP APIs.

**Architecture:** This cleanup is deletion-only. We remove files that import nonexistent modules or classes from the old AMP stack and then sweep the repository for dangling references so the remaining tree only points at the current manager-based environment APIs.

**Tech Stack:** Python, ripgrep, git-aware repository cleanup

---

### Task 1: Record the broken entrypoints

**Files:**
- Modify: `docs/plans/2026-03-12-remove-broken-legacy-entrypoints.md`

**Step 1: Identify files importing deleted AMP modules**

Run: `rg -n "amp_flat_env_cfg|amp_env import AmpEnv|UnitreeGo2FlatEnvCfg|tasks\\.locomotion\\.amp\\.utils_amp\\.motion_loader" source/legged_lab/test source/legged_lab/legged_lab/utils`

**Step 2: Confirm these files are not part of the current supported entrypoints**

Check current AMP entrypoints under `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/` and `source/legged_lab/legged_lab/envs/`.

### Task 2: Delete broken legacy files

**Files:**
- Delete: `source/legged_lab/legged_lab/utils/export_deploy_cfg.py`
- Delete: `source/legged_lab/test/test_amp_observations.py`
- Delete: `source/legged_lab/test/test_contact.py`
- Delete: `source/legged_lab/test/test_frame_transformer.py`
- Delete: `source/legged_lab/test/test_motion_data.py`
- Delete: `source/legged_lab/test/test_motion_loader.py`
- Delete: `source/legged_lab/test/test_motion_loader_2.py`
- Delete: `source/legged_lab/test/test_terrain_2.py`
- Delete: `source/legged_lab/test/test_ang_vel_from_quat_diff.py`

**Step 1: Remove files that depend on deleted APIs**

Delete only files whose imports or behavior still depend on the removed AMP stack.

**Step 2: Leave current supported runtime entrypoints untouched**

Do not modify `scripts/rsl_rl/*.py`, current env configs, or current gym registrations.

### Task 3: Verify no dangling references remain

**Files:**
- Verify: repository-wide references

**Step 1: Search for removed legacy import targets**

Run: `rg -n "amp_flat_env_cfg|amp_env import AmpEnv|UnitreeGo2FlatEnvCfg|tasks\\.locomotion\\.amp\\.utils_amp\\.motion_loader|obs\\[\\\"amp\\\"\\]" source/legged_lab scripts`

**Step 2: Inspect git status**

Run: `git status --short`

**Step 3: Report residual risks**

Call out any remaining obsolete runtime issues that are outside this deletion-only cleanup scope.
