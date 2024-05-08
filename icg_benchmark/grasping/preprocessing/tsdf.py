from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from icg_benchmark.simulator.perception import TSDFVolume, create_tsdf
from icg_benchmark.utils.timing.timer import Timer

from .base import ObsProcessor


@dataclass
class GigaObservation:
    tsdf: TSDFVolume
    tsdf_process: TSDFVolume
    camera_pose: np.ndarray


class GigaTSDFObservation(ObsProcessor[GigaObservation]):
    def __init__(
        self,
        lower_bounds=np.array([0, 0, 0]),
        upper_bounds=np.array([0, 0, 1]),
        tsdf_size=0.3,
        tsdf_res=40,
        tsdf_highres=60,
        noise_level: float = 0.0008,
    ) -> None:
        self.noise_level = noise_level
        self.tsdf_size = tsdf_size
        self.tsdf_res = tsdf_res
        self.tsdf_highres = tsdf_highres

    def __call__(
        self,
        depth_imgs: list[np.ndarray],
        intrinsic: list[float],
        extrinsics: list[list[float]],
        eye: np.ndarray,
        timer: Timer,
    ) -> GigaObservation | None:

        # add noise
        if self.noise_level > 0:
            # add gaussian noise 95% confident interval (-1.96,1.96)
            depth_imgs = np.array(
                [
                    d + np.random.normal(loc=0.0, scale=self.noise_level, size=d.shape).astype(d.dtype)
                    for d in depth_imgs
                ]
            )
        with timer["tsdf_integration"]:
            tsdf = create_tsdf(self.tsdf_size, self.tsdf_res, depth_imgs, intrinsic, extrinsics)

        with timer["tsdf_postproc"]:
            if self.tsdf_highres != self.tsdf_res:
                tsdf_highres = create_tsdf(self.tsdf_size, self.tsdf_highres, depth_imgs, intrinsic, extrinsics)
            else:
                tsdf_highres = tsdf
            state = GigaObservation(tsdf=tsdf, tsdf_process=tsdf_highres, camera_pose=eye)
            return state
