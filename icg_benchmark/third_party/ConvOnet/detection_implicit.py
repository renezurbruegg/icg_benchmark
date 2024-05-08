import time

import numpy as np
import torch
import trimesh
from scipy import ndimage

# from vgn import vis
from icg_benchmark.simulator.grasp import Grasp, from_voxel_coordinates
from icg_benchmark.simulator.transform import Rotation, Transform

from icg_benchmark.models.giga.networks import load_network
from icg_benchmark.grasping.planners.base import GraspPlannerModule
from icg_benchmark.grasping.preprocessing.tsdf import GigaTSDFObservation

from icg_benchmark.simulator.grasp import Grasp
from icg_benchmark.simulator.perception import *
from icg_benchmark.simulator.transform import Rotation, Transform

LOW_TH = 0.2  # 0.5 for giga


class VGNImplicit(object):
    def __init__(
        self,
        model_path,
        model_type,
        best=False,
        force_detection=False,
        qual_th=0.9,
        out_th=0.5,
        visualize=False,
        resolution=40,
        coll_check=False,
        **kwargs
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device, model_type=model_type)
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        self.coll_check = coll_check

        self.resolution = resolution
        x, y, z = torch.meshgrid(
            torch.linspace(
                start=-0.5,
                end=0.5 - 1.0 / self.resolution,
                steps=self.resolution,
            ),
            torch.linspace(
                start=-0.5,
                end=0.5 - 1.0 / self.resolution,
                steps=self.resolution,
            ),
            torch.linspace(
                start=-0.5,
                end=0.5 - 1.0 / self.resolution,
                steps=self.resolution,
            ),
        )
        # 1, self.resolution, self.resolution, self.resolution, 3
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)
        self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)

    def __call__(
        self,
        state,
        timer,
        scene_mesh=None,
        aff_kwargs={},
        return_reconstruction=False,
    ):
        if hasattr(state, "tsdf_process"):
            tsdf_process = state.tsdf_process
        else:
            tsdf_process = state.tsdf

        if isinstance(state.tsdf, np.ndarray):
            tsdf_vol = state.tsdf
            voxel_size = 0.3 / self.resolution
            size = 0.3
        else:
            tsdf_vol = state.tsdf.get_grid()
            voxel_size = tsdf_process.voxel_size
            tsdf_process = tsdf_process.get_grid()
            size = state.tsdf.size

        with timer["predict"]:
            tic = time.time()
            qual_vol, rot_vol, width_vol, embeddings = predict(
                tsdf_vol,
                self.pos,
                self.net,
                self.device,
                coll_check=self.coll_check,
            )

        with timer["process"]:
            qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
            rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
            width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))
            qual_vol, rot_vol, width_vol = process(tsdf_process, qual_vol, rot_vol, width_vol, out_th=self.out_th)

            qual_vol = bound(qual_vol, voxel_size)

            grasps, scores = select(
                qual_vol.copy(),
                self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(),
                rot_vol,
                width_vol,
                threshold=self.qual_th,
                force_detection=self.force_detection,
                max_filter_size=8 if self.visualize else 4,
            )

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        if return_reconstruction:
            print("caluclate reconstruction")

            from icg_benchmark.third_party.ConvOnet.generation import Generator3D

            generator = Generator3D(
                self.net,
                resolution0=16,
                upsampling_steps=3,
                device="cuda",
                threshold=0.5,
                input_type="pointcloud",
                padding=0,
            )

            tsdf_vol_ = torch.from_numpy(tsdf_vol).to("cuda")
            pred_mesh, _ = generator.generate_mesh({"inputs": tsdf_vol_})
            pred_mesh.vertices = (pred_mesh.vertices + 0.5) * 0.3

        new_grasps = []
        sc = []
        print("numbe graps", len(grasps))
        if len(grasps) > 0:
            if self.best:
                p = [np.arange(len(grasps))[0]]
            else:
                p = np.random.permutation(len(grasps))

            for s, g in zip(scores[p], grasps[p]):
                pose = g.pose
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                new_g = Grasp(pose, width)
                sc.append(s)
                # import open3d as o3d
                # scan = o3d.geometry.PointCloud()
                # scan.points = o3d.utility.Vector3dVector(new_g.get_fingertips())
                # # scan.points = o3d.utility.Vector3dVector(state.tsdf.get_pointcloud())
                # o3d.visualization.draw([state.pc, scan])
                new_grasps.append(new_g)
            # scores = scores[p]
        grasps = new_grasps
        if len(grasps) == 0:
            return None
        # make se3 pose
        pose = torch.eye(4, device=self.device).unsqueeze(0)
        pose = np.stack([g.pose.as_matrix() for g in grasps])  # grasps[0].pose.as_matrix()[None, ...]
        # pose[:, :3, :3] = grasps[0].pose. .view(3, 3)
        # Convert to edge grasp gripper convention
        pose[:, :3, 3] = grasps[0].pose.translation - pose[:, :3, :3] @ np.array([0, 0, 0.022])
        if return_reconstruction:
            return (pose, None), pred_mesh
        return (pose, None)


def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    # avoid grasp out of bound [0.02  0.02  0.055]
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    qual_vol[:x_lim] = 0.0
    qual_vol[-x_lim:] = 0.0
    qual_vol[:, :y_lim] = 0.0
    qual_vol[:, -y_lim:] = 0.0
    qual_vol[:, :, :z_lim] = 0.0
    return qual_vol


def predict(tsdf_vol, pos, net, device, coll_check=False):
    assert tsdf_vol.shape == (1, 40, 40, 40)

    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).to(device)

    # forward pass
    with torch.no_grad():
        c = net.encode_inputs(tsdf_vol)
        # feature = self.query_feature(p, c)
        # qual, rot, width = self.decode_feature(p, feature)
        qual_vol, rot_vol, width_vol = net.decode(pos, c)
    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol, c


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=0.033,
    max_width=0.233,
    out_th=0.5,
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(qual_vol, sigma=gaussian_filter_sigma, mode="nearest")

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(outside_voxels, iterations=2, mask=np.logical_not(inside_voxels))
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(
    qual_vol,
    center_vol,
    rot_vol,
    width_vol,
    threshold=0.90,
    max_filter_size=4,
    force_detection=False,
):
    best_only = False
    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        best_only = True
    else:
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    if best_only and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]

    return sorted_grasps, sorted_scores


def select_index(qual_vol, center_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    # pos = np.array([i, j, k], dtype=np.float64)
    pos = center_vol[i, j, k].numpy()
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score
