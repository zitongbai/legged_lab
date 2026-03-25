from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def motion_data_finish(env: ManagerBasedAnimationEnv) -> torch.Tensor:
    flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for term_name in env.animation_manager.active_terms:
        term: AnimationTerm = env.animation_manager.get_term(term_name)
        term_flag = term.motion_fetch_time[:, -1] >= term.motion_durations
        flag = flag | term_flag
    return flag
