import argparse
import time
from typing import Optional

import numpy as np
import open3d as o3d

from icg_benchmark.simulator.perception import create_tsdf
from icg_benchmark.utils.timing.timer import Timer

from .base import ObsProcessor


class EdgeGraspObservation(ObsProcessor[o3d.geometry.PointCloud]):
    def __init__(
        self,
        voxelization_size=0.0045,
        lower_bounds=np.array([0, 0, 0]),
        upper_bounds=np.array([0, 0, 1]),
        tsdf_size: float = 0.3,
        tsdf_resolution: int = 180,
        with_table: bool = False,
        noise_level: float = 0.0008,
    ) -> None:
        self.tsdf_resolution = tsdf_resolution
        self.with_table = with_table
        self.noise_level = noise_level
        self.tsdf_size = tsdf_size
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.voxelization_size = voxelization_size

    def __call__(
        self,
        depth_imgs: list[np.ndarray],
        intrinsic: list[float],
        extrinsics: list[list[float]],
        eye: np.ndarray,
        timer: Timer,
    ) -> o3d.geometry.PointCloud | None:

        # Fuse image into tsdf
        with timer["tsdf_integration"]:
            tsdf = create_tsdf(self.tsdf_size, 180, depth_imgs, intrinsic, extrinsics)
            pc = tsdf.get_cloud()

        with timer["tsdf_postproc"]:
            # crop surface and borders from point cloud
            if self.with_table:
                bounding_box = o3d.geometry.AxisAlignedBoundingBox(
                    self.lower_bounds - np.array([0, 0, 0.05]), self.upper_bounds
                )
            else:
                bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower_bounds, self.upper_bounds)

            pc = pc.crop(bounding_box)
            if pc.is_empty():
                return None

            if self.noise_level > 0:
                vertices = np.asarray(pc.points)
                # add gaussian noise 95% confident interval (-1.96,1.96)
                vertices = vertices + np.random.normal(loc=0.0, scale=self.noise_level, size=(len(vertices), 3))
                pc.points = o3d.utility.Vector3dVector(vertices)

            vertices = np.asarray(pc.points)

            if len(vertices) < 100:
                print("point cloud<100, skipping scene")
                return None

            time0 = time.time()
            pc, ind = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
            pc, ind = pc.remove_radius_outlier(nb_points=30, radius=0.03)
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
            pc.orient_normals_consistent_tangent_plane(30)
            # orient the normals direction
            # convert to o3d convention
            # eye[0] = -eye[0]
            # eye[1] = -eye[1]

            # eye_pc = o3d.geometry.PointCloud()
            # eye_pc.points = o3d.utility.Vector3dVector(eye + np.random.randn(1000,3)*0.001)

            pc.orient_normals_towards_camera_location(camera_location=eye)
            # o3d.visualization.draw_geometries([pc, eye_pc], point_show_normal=True)

            normals = np.asarray(pc.normals)
            vertices = np.asarray(pc.points)
            if len(vertices) < 100:
                print("point cloud<100, skipping scene")
                return None

            pc = pc.voxel_down_sample(voxel_size=self.voxelization_size)
            return pc


class ICGNetObservation(ObsProcessor[o3d.geometry.PointCloud]):
    def __init__(
        self,
        voxelization_size=0.0045,
        lower_bounds=np.array([0, 0, 0]),
        upper_bounds=np.array([0, 0, 1]),
        tsdf_size: float = 0.3,
        tsdf_resolution: int = 180,
        with_table: bool = False,
        noise_level: float = 0.0008,
    ) -> None:
        self.tsdf_resolution = tsdf_resolution
        self.with_table = with_table
        self.noise_level = noise_level
        self.tsdf_size = tsdf_size
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.voxelization_size = voxelization_size

    def __call__(
        self,
        depth_imgs: list[np.ndarray],
        intrinsic: list[float],
        extrinsics: list[list[float]],
        eye: np.ndarray,
        timer: Timer,
    ) -> o3d.geometry.PointCloud | None:
        # reconstrct point cloud using a subset of the images
        with timer["tsdf_integration"]:
            tsdf = create_tsdf(self.tsdf_size, 180, depth_imgs, intrinsic, extrinsics)
            pc = tsdf.get_cloud()

        with timer["tsdf_postproc"]:
            # crop surface and borders from point cloud
            if self.with_table:
                bounding_box = o3d.geometry.AxisAlignedBoundingBox(
                    self.lower_bounds - np.array([0, 0, 0.05]), self.upper_bounds
                )

            else:
                bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower_bounds, self.upper_bounds)

            pc = pc.crop(bounding_box)
            if pc.is_empty():
                return None

            if self.noise_level > 0:
                vertices = np.asarray(pc.points)
                # add gaussian noise 95% confident interval (-1.96,1.96)
                vertices = vertices + np.random.normal(loc=0.0, scale=self.noise_level, size=(len(vertices), 3))
                pc.points = o3d.utility.Vector3dVector(vertices)

            vertices = np.asarray(pc.points)

            if len(vertices) < 100:
                print("point cloud<100, skipping scene")
                return None

            pc, ind = pc.remove_radius_outlier(nb_points=20, radius=0.02)
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
            pc.orient_normals_towards_camera_location(camera_location=eye)

            # normals = np.asarray(pc.normals)
            vertices = np.asarray(pc.points)
            if len(vertices) < 100:
                print("point cloud<100, skipping scene")
                return None

            pc = pc.voxel_down_sample(voxel_size=self.voxelization_size)
            return pc
