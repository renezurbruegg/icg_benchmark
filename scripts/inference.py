from icg_benchmark.grasping.eval import GraspEvaluator

from icg_benchmark.grasping.planners import EdgeGraspPlanner
from icg_benchmark.grasping.preprocessing import ObsProcessor
from icg_benchmark.utils.timing.timer import Timer


import argparse
import datetime
import os
import numpy as np

import open3d as o3d

from icg_benchmark.models.edge_grasp.edge_grasper import EdgeGrasper


import trimesh
import numpy as np
from trimesh import transformations
from scipy.spatial.transform import Rotation

def create_gripper_marker(
    rot,
    color=[0, 0, 255],
    tube_radius=0.002,
    sections=6,
    center_offset=[0, 0, 0],
    width=0.08,
):
    finger_length = 0.05  # 5cm finger length
    gripper_finger_left = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [width / 2 + tube_radius, 0, finger_length],
            [width / 2 + tube_radius, 0, 0],
        ],
    )
    gripper_finger_right = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [-(width / 2 + tube_radius), 0, finger_length],
            [-(width / 2 + tube_radius), 0, 0],
        ],
    )

    # handle
    gripper_handle = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, -0.07], [0, 0, 0]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-(width / 2 + tube_radius), 0, 0], [width / 2 + tube_radius, 0, 0]],
    )

    tmp = trimesh.util.concatenate(
        [
            gripper_handle,
            cb2,
            gripper_finger_right,
            gripper_finger_left,
        ]
    )
    tmp.visual.face_colors = color

    tmp.apply_transform(rot)

    return tmp


class PointcloudPreprocesor(ObsProcessor[o3d.geometry.PointCloud]):
    def __init__(
        self,
        voxelization_size=0.0045,
        lower_bounds=np.array([0, 0, 0]),
        upper_bounds=np.array([1, 1, 1]),
        tsdf_size: float = 0.3,
        tsdf_resolution: int = 180,
    ) -> None:
        self.tsdf_resolution = tsdf_resolution
        self.tsdf_size = tsdf_size
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.voxelization_size = voxelization_size

    def __call__(
        self,
        pc: o3d.geometry.PointCloud,
    ) -> o3d.geometry.PointCloud | None:

        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower_bounds, self.upper_bounds)

        pc = pc.crop(bounding_box)
        if pc.is_empty():
            return None

        pc, ind = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
        pc, ind = pc.remove_radius_outlier(nb_points=30, radius=0.03)
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))

        normals = np.asarray(pc.normals)
        vertices = np.asarray(pc.points)

        if len(vertices) < 100:
            print("point cloud<100, skipping scene")
            return None

        pc = pc.voxel_down_sample(voxel_size=self.voxelization_size)
        return pc


def main():
    timer = Timer()

    # Define the grasp planner
    planner = EdgeGraspPlanner(
        EdgeGrasper(device="cuda", root_dir="data/edge_grasp_net_pretrained_para", load=180),
        confidence_th=0.5,
        return_best_score=False,
    )

    # Define the pre-processing function for the point cloud
    # (de-noise, crop, downsample, estimate normals, etc.)
    preprocessor = PointcloudPreprocesor()


    pc = o3d.io.read_point_cloud("scripts/scan.ply")
    pc = preprocessor(pc)

    # Dummy function, which when called, returns the point cloud.
    # Needs to be passed to the planner.
    def get_pc(*args, **kwargs):
        return pc
        
    predictions = planner(get_pc, timer)

    # Visualize the predictions
    grasp_meshes = []
    for grasp in predictions[0]:
        # Predicted gripper position (1x3 vector)
        pos = grasp[:-1, -1]
        # predicted gripper orientation (3x3 rotation matrix)
        rot = grasp[:-1, :-1]
        marker = create_gripper_marker(grasp, color=[0, 0, 255])
        grasp_meshes.append(marker.as_open3d)

    o3d.visualization.draw_geometries([pc, *grasp_meshes])



if __name__ == "__main__":
    main()