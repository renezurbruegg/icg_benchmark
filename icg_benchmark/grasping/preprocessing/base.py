from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np

from icg_benchmark.utils.timing.timer import Timer

ObsType = TypeVar("ObsType")


class ObsProcessor(Generic[ObsType]):

    def __call__(
        self,
        depth_imgs: list[np.ndarray],
        intrinsic: list[float],
        extrinsics: list[list[float]],
        eye: np.ndarray,
        timer: Timer,
    ) -> ObsType | None:
        raise NotImplementedError("Subclasses must implement this method")
