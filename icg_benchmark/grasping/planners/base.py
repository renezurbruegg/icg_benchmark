from __future__ import annotations

from typing import Callable, Generic, TypeVar

import torch
import torch.nn as nn

from icg_benchmark.utils.timing.timer import Timer

ObsType = TypeVar("ObsType")


class GraspPlannerModule(nn.Module, Generic[ObsType]):
    """Simple interface for grasp planners."""

    def forward(
        self, observation_cb: Callable[[Timer | None], ObsType | None], timer: Timer
    ) -> tuple[torch.Tensor, float | None] | None:
        """Calculates feasible grasps for a given observation."""
        raise NotImplementedError("Implement this method in the subclass.")

    def grasp_cb(self, success: bool) -> None:
        """Callback function after executing a grasp

        Args:
            success (bool): Whether the grasp was successful.
        """
        pass

    def reset(self) -> None:
        """Reset the planner.

        This method gets called after the scene has been reset.
        """
        pass
