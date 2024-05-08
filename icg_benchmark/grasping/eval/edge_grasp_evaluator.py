"""Evaluator that follows the evaluation pipeline from EdgeGraspNet."""

import copy
import json
import os
from math import cos, sin
from typing import Callable

import numpy as np
import torch
import tqdm

from icg_benchmark.grasping.planners import GraspPlannerModule
from icg_benchmark.grasping.preprocessing import EdgeGraspObservation, ObsProcessor
from icg_benchmark.simulator.grasp import Grasp, Label
from icg_benchmark.simulator.perception import camera_on_sphere
from icg_benchmark.simulator.simulation_clutter_bandit import ClutterRemovalSim
from icg_benchmark.simulator.transform import Rotation, Transform
from icg_benchmark.utils.seed import seed_everything
from icg_benchmark.utils.timing.timer import Timer

from .base import RunStatistics


class GraspEvaluator:
    """Simple Grasp Evaluator that follows the evaluation pipeline from EdgeGraspNet.

    Args:
        scene (str, optional): The scene to use. Defaults to "packed".
            Select from ["packed", "pile"]
        object_set (str, optional): The object set to use. Defaults to "packed/test".
        show_gui (bool, optional): Whether to show the pybullet GUI.
            Defaults to False.
        rand (bool, optional): Whether to use random object orientations.
            Defaults to True.
        verbose (bool, optional): Whether to show progress bars. Defaults to True.
        sort_file_list (bool, optional): Whether to sort the file list in the
            ClutterRemovelSim, to ensure more reproducibility. Defaults to False.
        preproc (ObsProcessor): Callable that returns the observation processor to use.
            Defaults to EdgeGraspObservation.

    """

    def __init__(
        self,
        scene: str = "packed",
        object_set: str = "packed/test",
        show_gui: bool = False,
        rand: bool = True,
        verbose: bool = True,
        sort_file_list: bool = False,
        preproc: Callable[[float, float], ObsProcessor] = lambda lower, upper: EdgeGraspObservation(lower, upper),
    ) -> None:

        # Internal simulator from VGN
        self.sim = ClutterRemovalSim(
            scene,
            object_set,
            gui=show_gui,
            rand=rand,
            sort_file_list=sort_file_list,
        )
        self.verbose = verbose

        # Calculate current run statistics
        self.current_run = RunStatistics(
            sucesses=0,
            tries=0,
            total_objects=0,
            skips=0,
            num_imgs=0,
            gripper_collisions=0,
            object_object_collisions=0,
            all_collisions=0,
        )

        self.preproc = preproc(lower_bounds=self.sim.lower, upper_bounds=self.sim.upper)
        # Timing module to keep track of execution times
        self.timer = Timer()

    def evaluate_grasps(self, poses: torch.Tensor, width_pre: np.ndarray | None = None) -> bool:
        """Evaluate a set of grasps.

        Args:
            poses (torch.Tensor): The grasp poses to evaluate.
            width_pre (np.ndarray, optional): The grasp widths to evaluate.
                if None, the maximum gripper width is used. Defaults to None.
        """
        outcomes, widths, describtions = [], [], []
        quats, translations = [], []
        for i in range(len(poses)):
            pose = poses[i, :, :]
            dof_6 = Transform.from_matrix(pose)
            # decompose the quat
            quat = dof_6.rotation.as_quat()
            translation = dof_6.translation
            if width_pre is not None:
                width = width_pre[i]
            else:
                width = self.sim.gripper.max_opening_width

            candidate = Grasp(dof_6, width=width)
            (
                outcome,
                width,
                describtion,
                collision,
            ) = self.sim.execute_grasp_quick(
                candidate,
                allow_contact=True,
                remove=True,
                return_collision=True,
            )

            if len(collision) > 0:
                self.current_run["all_collisions"] += 1
                gripper = [c for c in collision if c.bodyA.name == "panda" or c.bodyB.name == "panda"]
                if len(gripper) > 0:
                    self.current_run["gripper_collisions"] += 1

            outcomes.append(outcome)
            widths.append(width)
            describtions.append(describtion)
            quats.append(quat)
            translations.append(translation)
        successes = (np.asarray(outcomes) == Label.SUCCESS).astype(int)
        return successes

    def render_images(
        self, n_views: int = 1, static_view_fnc: Callable[[], tuple[float, float, float]] | None = None
    ) -> tuple[list[np.ndarray], list[list[float]], list[list[float]]]:
        """Render n_views images of the current scene.

        Args:
            n_views (int, optional): The number of views to render. Defaults to 1.
            static_view_fnc (Callable[[], tuple[float, float, float]], optional): A function that returns a viewpoint
                as spherical coordinates.
                If provided, the random view generation is skipped. Defaults to None.
        """
        self.current_run["num_imgs"] += n_views

        height, width = (
            self.sim.camera.intrinsic.height,
            self.sim.camera.intrinsic.width,
        )
        origin = Transform(
            Rotation.identity(),
            np.r_[self.sim.size / 2, self.sim.size / 2, 0.0 + 0.25],
        )
        extrinsics = np.empty((n_views, 7), np.float32)
        depth_imgs = np.empty((n_views, height, width), np.float32)
        for i in range(n_views):
            r = np.random.uniform(2, 2.5) * self.sim.size
            theta = np.random.uniform(np.pi / 4, np.pi / 3)
            phi = np.random.uniform(0.0, np.pi)

            if static_view_fnc:
                r, theta, phi = static_view_fnc()

            extrinsic = camera_on_sphere(origin, r, theta, phi)

            depth_img = self.sim.camera.render(extrinsic)[1]
            extrinsics[i] = extrinsic.to_list()
            depth_imgs[i] = depth_img
            eye = np.r_[
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta),
            ]
            cam_pos = eye + origin.translation
        return depth_imgs, extrinsics, cam_pos

    def eval_method(
        self,
        grasp_planner: GraspPlannerModule,
        object_count: int = 5,
        num_runs: int = 4,
        num_rounds: int = 100,
        num_views: int = 1,
        static_view_fnc: Callable[[], tuple[float, float, float]] | None = None,
        log_file: str | None = None,
    ) -> list[RunStatistics]:
        """Main entry point to evaluate a grasp planner.

        Args:
            grasp_planner (GraspPlannerModule): The grasp planner to evaluate.
            object_count (int, optional): The number of objects to spawn. Defaults to 5.
            num_runs (int, optional): The number of runs to average over. Defaults to 4.
            num_rounds (int, optional): The number of rounds per run. Defaults to 100.
            num_views (int, optional): The number of views per round. Defaults to 1.
            static_view_fnc (Callable[[], tuple[float, float, float]], optional): A function that returns a viewpoint
                as spherical coordinates.
                If provided, the random view generation is skipped. Defaults to None.
            log_file (str, optional): The file to log the results to. Defaults to None, which does not log the results.
            Returns:
                list[RunStatistics]: A list of RunStatistics objects, one for each run.
        """
        record = []

        for run in range(num_runs):
            print(f"[Simulator]: Starting run {run + 1} of {num_runs}")

            seed_everything(run + 1)

            cnt = 0
            success = 0
            left_objs = 0
            total_objs = 0
            cons_fail = 0

            self.current_run = RunStatistics(
                sucesses=0,
                tries=0,
                total_objects=0,
                skips=0,
                num_imgs=0,
                gripper_collisions=0,
                object_object_collisions=0,
                all_collisions=0,
            )

            pbar = tqdm.tqdm(range(num_rounds), disable=not self.verbose)
            for _ in pbar:
                _ = (
                    np.random.poisson(4) + 1
                )  # this needs to stay to be comparable to edgegrasp (changes random state of numpy)

                self.sim.reset(object_count)
                if hasattr(grasp_planner, "reset"):
                    grasp_planner.reset()

                self.current_run["total_objects"] += self.sim.num_objects

                ###
                total_objs += self.sim.num_objects
                consecutive_failures = 1
                last_label = None
                trial_id = -1
                ###
                skip_time = 0
                while self.sim.num_objects > 0 and consecutive_failures < 2 and skip_time < 3:
                    trial_id += 1

                    def get_img(timer):
                        depth_imgs, extrinsics, eye = self.render_images(num_views, static_view_fnc=static_view_fnc)
                        return self.preproc(depth_imgs, self.sim.camera.intrinsic, extrinsics, eye, timer)

                    out = grasp_planner(get_img, self.timer)

                    if out is None:
                        print("Skipping Scene...")
                        self.current_run["skips"] += 1
                        skip_time += 1
                        continue
                    else:
                        grasp_pose, width = out
                        success_num = self.evaluate_grasps(grasp_pose, width)
                        self.current_run["tries"] += 1

                        cnt += 1
                        label = success_num[0]

                        if label != Label.FAILURE:
                            self.current_run["sucesses"] += 1
                            success += 1

                        if last_label == Label.FAILURE and label == Label.FAILURE:
                            consecutive_failures += 1
                        else:
                            consecutive_failures = 1

                        if consecutive_failures >= 2:
                            cons_fail += 1
                        last_label = label

                    if hasattr(grasp_planner, "grasp_cb"):
                        grasp_planner.grasp_cb(label != Label.FAILURE)

                left_objs += self.sim.num_objects

                success_rate = 100.0 * self.current_run["sucesses"] / self.current_run["tries"]
                declutter_rate = 100.0 * self.current_run["sucesses"] / self.current_run["total_objects"]
                pbar.set_description(f"GSR: {success_rate:.2f}%, DCR: {declutter_rate:.2f}%")
                self.current_run["timings"] = self.timer.get_stats()

            record.append(copy.deepcopy(self.current_run))

        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            gsr = np.array([d["sucesses"] / d["tries"] for d in record])
            dr = np.array([d["sucesses"] / d["total_objects"] for d in record])
            all_results = {
                "gsr": np.mean(gsr),
                "gsr_std": np.std(gsr),
                "dr": np.mean(dr),
                "dr_std": np.std(dr),
                "collisions": np.mean([d["all_collisions"] for d in record]),
                "collisions_std": np.std([d["all_collisions"] for d in record]),
                "collisions_gripper": np.mean([d["gripper_collisions"] for d in record]),
                "collisions_gripper_std": np.std([d["gripper_collisions"] for d in record]),
                "all_results": record,
            }
            json.dump(all_results, open(log_file, "w"))

        return record
