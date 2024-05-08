from typing import Callable

import numpy as np
import open3d as o3d
import torch
try:
    from icg_net.typing import ModelPredOut,  SceneEmbedding
    from icg_net.vis.visualizer import GraspVisualizer
    from icg_net import ICGNetModule
except ImportError as e:
    print("Failed to import icg_net dependencies. Make sure you have the icg_net package installed.", str(e))

from termcolor import colored

from icg_benchmark.simulator.io_smi import *
from icg_benchmark.simulator.utility import FarthestSamplerTorch
from icg_benchmark.utils.timing.timer import Timer

from .base import GraspPlannerModule


class ICGNetPlanner(GraspPlannerModule[o3d.geometry.PointCloud]):
    def __init__(
        self,
        model: ICGNetModule,
        device: str = "cuda",
        confidence_th: float = 0.4,
        resample: bool = False,
        visualize: bool = False,
        latent_imagination: bool = False,
        use_fps: bool = False,
    ) -> None:
        """Grasp planner using ICGNet.

        Args:
            model: Model to use for inference
            device: Device to run inference on
            confidence_th: Confidence threshold for selecting grasps
            resample: Resample grasps, if True and no feasible grasps are found, resample grasps with imagined surface
                      points
            visualize: Visualize grasps
            latent_imagination: If true, does not re-calculate the embeddings for the pointcloud, but just remove an
                                embedding from the queries to predict the next grasps.
                                If this is true, a full scene could be cleared with only one scan.
            use_fps: Use farthest point sampling to sample grasp proposals.
        """
        super().__init__()
        self.device = device
        self.confidence_th = confidence_th
        self.model = model
        self.resample = resample
        self.visualize = visualize
        self.latent_imagination = latent_imagination

        self.cloud = None
        self.recapture = False
        self.last_id = None
        self.embeddings = None
        self.query_mask = [[]]
        self.use_fps = use_fps

    def grasp_cb(self, success):
        if not self.latent_imagination:
            return self.reset()

        print("  ")
        print("  ")
        print("  ")
        if success:
            print("==[Latent Imageination] Grasp successfull. Going to mark {} as grasped".format(self.last_id))
            if self.query_mask is None:
                self.query_mask = [[]]

            mid = self.last_id + 0
            for i in self.query_mask[0]:
                if i <= mid:
                    print("Increasing id by one since {} is smaller than {}".format(i, mid))
                    self.last_id += 1  # shift this id by one

            self.query_mask[0].append(self.last_id)
            print("New query mask", self.query_mask)
        else:
            print("Failed grasp. Going to request recapture!")

            self.reset()
        print("  ")
        print("  ")
        print("  ")
        print("  ")

    def reset(self):
        self.query_mask = [[]]
        self.cloud = None
        self.last_id = None
        self.embeddings = None

    def forward(self, observation_cb: Callable[[Timer | None], o3d.geometry.PointCloud | None], timer: Timer):
        if not self.latent_imagination:
            self.reset()

        with timer["preprocess"]:
            if self.cloud is None or self.recapture:
                self.replay = False
                pc = observation_cb(timer)
                if pc is None:
                    return None
                self.cloud = pc
            pc = self.cloud

            pos = torch.from_numpy(np.asarray(pc.points)).to(torch.float32).to(self.device)
            normals = torch.from_numpy(np.asarray(pc.normals)).to(torch.float32).to(self.device)

            if self.use_fps and len(pos) > 2048:
                fps_sample = FarthestSamplerTorch()

                _, sample = fps_sample(pos, min(2048, len(pos)))
                sample = torch.as_tensor(sample).to(torch.long).reshape(-1).to(self.device)
                pos_grasp = pos[sample, :]
                normals_grasp = normals[sample, :]

            else:
                pc = pc.voxel_down_sample(voxel_size=0.002)

                pos_grasp = torch.from_numpy(np.asarray(pc.points)).to(torch.float32).to(self.device)
                normals_grasp = torch.from_numpy(np.asarray(pc.normals)).to(torch.float32).to(self.device)

        if len(pos) == 0 or len(pos_grasp) == 0:
            self.reset()
            return None

        if self.embeddings is None:
            with timer["inference"]:
                output = self.model(
                    pos,
                    None,
                    normals,
                    pos_grasp,
                    normals_grasp,
                    return_meshes=False,
                    return_scene_grasps=True,
                    return_object_grasps=False,
                    resample=self.resample,
                )
            self.embeddings = output
        else:
            output = self.embeddings
            # filter out grasped instances (1 x n_inst x query_dim)
            latent_grasp_inst = output.embedding.scene_grasps
            instances_to_keep = [i for i in range(latent_grasp_inst.shape[1]) if i not in self.query_mask[0]]
            print("keeping instances,", instances_to_keep)

            latent_grasp_inst = latent_grasp_inst[:, instances_to_keep, :]
            shape_inst = output.embedding.shape[instances_to_keep, :]

            pos_encodings = output.embedding.pos_encodings[instances_to_keep, :]
            recons = [r for idx, r in enumerate(output.reconstructions) if idx in instances_to_keep]

            new_embeddings = SceneEmbedding(
                object_grasps=output.embedding.object_grasps,
                scene_grasps=latent_grasp_inst,
                shape=shape_inst,
                pos_encodings=pos_encodings,
                class_labels=output.embedding.class_labels,
                voxelized_pc=output.embedding.voxelized_pc,
                pointwise_labels=output.embedding.pointwise_labels,
                voxel_assignement=output.embedding.voxel_assignement,
                semseg_points=output.embedding.semseg_points,
                semseg_latents=output.embedding.semseg_latents,
                semseg=output.embedding.semseg,
            )

            scene_grasp_poses, obj_grasp_poses = self.model._get_grasps(
                pos_grasp, normals_grasp, new_embeddings, True, False
            )
            output = ModelPredOut(
                embedding=new_embeddings,
                class_predictions=output.class_predictions,
                scene_grasp_poses=scene_grasp_poses,
                object_grasp_poses=output.object_grasp_poses,
                reconstructions=recons,
            )

        with timer["postprocess"]:
            # Filter grasps
            grasps, scores = [], []
            if len(output.scene_grasp_poses) == 0:
                return None

            ori, ctc, score, width, instance = output.scene_grasp_poses
            mask = score > self.confidence_th
            if not mask.any():
                print(
                    colored(
                        "no candidates with high score" + f"{str(score.max()) if len(score) > 0 else 'No pts'}",
                        "yellow",
                    )
                )
                return None

            ori = ori[mask]
            ctc = ctc[mask]
            score = score[mask]
            width = width[mask]
            instance = instance[mask]

            best_grasps = score.argmax()
            self.last_id = instance[best_grasps].item()
            # make se3 pose
            pose = torch.eye(4, device=self.device).unsqueeze(0)
            pose[:, :3, :3] = ori[best_grasps].view(3, 3)
            # Convert to edge grasp gripper convention
            pose[:, :3, 3] = ctc[best_grasps] - pose[:, :3, :3] @ torch.tensor([0, 0, 0.022]).to(self.device)
            return (pose.cpu().numpy(), None)
