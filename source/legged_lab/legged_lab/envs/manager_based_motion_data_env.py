from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

from legged_lab.managers import MotionDataManager

from .manager_based_motion_data_env_cfg import ManagerBasedMotionDataEnvCfg


class ManagerBasedMotionDataEnv(ManagerBasedRLEnv):
    """Manager-based RL environment with a motion data manager and no animation manager."""

    cfg: ManagerBasedMotionDataEnvCfg

    def __init__(self, cfg: ManagerBasedMotionDataEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def load_managers(self):
        self.motion_data_manager = MotionDataManager(self.cfg.motion_data, self)
        print("[INFO] Motion Data Manager: ", self.motion_data_manager)
        super().load_managers()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        # Command resets may write reference root/joint states directly to sim.
        # Refresh articulation buffers before post-reset observations are computed.
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)

        for term in self.command_manager._terms.values():
            if hasattr(term, "_update_reference_state"):
                term._update_reference_state(env_ids)
