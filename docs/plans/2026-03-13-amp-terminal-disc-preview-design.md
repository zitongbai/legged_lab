# AMP Terminal Discriminator Observation Design

**Problem**

`ManagerBasedRLEnv.step()` in Isaac Lab computes task rewards and termination flags before reset, but returns observations after resetting terminated environments. This is correct for standard PPO usage, but the current AMP stack in this repository consumes the returned discriminator observation group (`disc`) to compute style rewards and populate discriminator replay buffers in `PPOAMP.process_env_step()`. For terminated environments, the returned `disc` observation therefore belongs to the next episode instead of the terminal transition.

**Goal**

Preserve the default Isaac Lab observation semantics for policy and critic observations while providing AMP with a correct terminal discriminator observation for terminated environments, without modifying the Isaac Lab source tree.

**Chosen Approach**

Add a local `ObservationManager` subclass in `legged_lab` that provides a non-mutating preview API for observation groups with history. The new API computes the observation group as if the current raw observation had been appended to the history buffer, but it does not modify the internal history buffer state.

`ManagerBasedAmpEnv` will use this preview API immediately before resetting terminated environments to compute a terminal `disc` snapshot and export it through `extras["terminal_disc_obs"]`. The main returned observation dictionary will keep Isaac Lab's default post-reset semantics.

`PPOAMP.process_env_step()` in the local `rsl_rl` checkout will consume `extras["terminal_disc_obs"]` and replace the discriminator observation for `dones == True` before computing style rewards and appending policy-side discriminator samples to its replay buffer.

**Alternatives Considered**

1. Override the returned `obs["disc"]` directly in `ManagerBasedAmpEnv.step()`
   This would work, but it mixes pre-reset and post-reset semantics inside the primary observation dictionary. That makes the environment contract harder to reason about and is less suitable for an upstreamable abstraction.

2. Reconstruct terminal discriminator observations directly inside `PPOAMP`
   This is not viable because the algorithm only receives the post-reset observation and does not have access to the observation manager history buffers required to reconstruct the correct terminal history window.

3. Patch Isaac Lab's `ObservationManager`
   This would likely be the cleanest long-term upstream change, but it violates the current repository constraint of not editing the local Isaac Lab source tree.

**Architecture**

1. Add a local observation manager subclass, reusing Isaac Lab's `ObservationGroupCfg` and `ObservationTermCfg`.
2. Override `ManagerBasedAmpEnv.load_managers()` to instantiate the local observation manager while otherwise preserving Isaac Lab manager initialization order and logging behavior.
3. Add a terminal discriminator preview step in `ManagerBasedAmpEnv.step()` between task reward computation and environment reset.
4. Update local `rsl_rl` AMP processing so that terminal discriminator observations are used only for terminated environments.

**Testing Strategy**

Tests should follow Isaac Lab's own testing style:

- Manager-level tests modeled after `source/isaaclab/test/managers/test_observation_manager.py`
- Environment-level tests modeled after Isaac Lab headless `AppLauncher` tests
- Minimal AMP algorithm tests in the local `rsl_rl` checkout to verify discriminator replacement behavior without requiring a full training run

The most important assertions are:

- previewing a group with history does not mutate the underlying history buffer
- previewing a group with history includes the current raw observation as the newest frame
- `ManagerBasedAmpEnv.step()` exports `extras["terminal_disc_obs"]` for terminated environments
- `PPOAMP.process_env_step()` uses `terminal_disc_obs` for style reward computation and discriminator replay buffer insertion only when `dones` is true

**Risks**

- Re-implementing observation-group post-processing in the subclass must stay behaviorally aligned with Isaac Lab's `ObservationManager.compute_group()`
- Overriding `ManagerBasedAmpEnv.load_managers()` must preserve the same manager construction order as the Isaac Lab base classes
- Future Isaac Lab upgrades may require re-syncing the subclass if the upstream observation manager changes materially
