from icg_benchmark.grasping.eval import GraspEvaluator

from icg_benchmark.grasping.planners import ICGNetPlanner
from icg_benchmark.grasping.preprocessing import ICGNetObservation
from icg_benchmark.grasping.view_samplers import top_down_view

import argparse
import datetime
import os

from icg_net import ICGNetModule


def get_model(config, device, args, graps_ori=6):
    return ICGNetModule(
        config=config,
        device=device,
        grasp_each_object=True,
        n_grasps=8192,
        n_grasp_pred_orientations=graps_ori,
        gripper_offset=0.0,
        gripper_offset_perc=10.5,
        max_gripper_width=0.08,
        full_width=args.full_width,
        coll_checks=not args.no_coll_checks,
    ).eval()


def main(args):
    def preproc(**kwargs):
        return ICGNetObservation(with_table=args.with_table, **kwargs)

    evaluator = GraspEvaluator(
        scene=args.scene, object_set=args.object_set, show_gui=args.sim_gui, rand=args.rand, preproc=preproc
    )
    model = get_model(args.config, args.device, args, args.n_ori)
    planner = ICGNetPlanner(
        model,
        confidence_th=args.th,
        resample=args.resample,
        visualize=args.vis,
        latent_imagination=args.latent_replay,
        use_fps=args.fps,
    )

    log_file = args.name + "_" + args.scene + "_" + args.object_set.replace("/", "_")
    log_file += datetime.datetime.now().strftime("_%m_%d_%H_%M_%S") + ".json"

    evaluator.eval_method(planner, static_view_fnc=top_down_view if args.top_down else None, log_file=os.path.join(args.logdir, log_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument("--scene", type=str, choices=["pile", "packed", "obj", "egad"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim-gui", action="store_true", default=False)
    parser.add_argument("--rand", action="store_true", default=True, help="Randomize the scene, by loading objects at random poses.")
    parser.add_argument("--th", type=float, default=0.4, help="Threshold for the ICGNet planner.")

    # ICGNet Arguments
    parser.add_argument("--top-down", action="store_true", default=False)
    parser.add_argument("--with_table", action="store_true", default=False)
    parser.add_argument("--resample", action="store_true", default=False)
    parser.add_argument("--name", type=str, default="icgnet")
    parser.add_argument("--full_width", action="store_true", default=True, help="Use full gripper width instead of the predicted width from the network.")
    parser.add_argument("--latent_replay", action="store_true", default=False, help="Use latent replay for the ICGNet planner.")
    parser.add_argument("--vis", action="store_true", default=False, help="Visualize the ICGNet planner.")
    parser.add_argument("--no-coll-checks", action="store_true", default=False, help="Disable internal collision checks for the ICGNet planner.")
    parser.add_argument("--fps", action="store_true", default=False, help="Use furthest point sampling for the ICGNet planner to find grasp candidates.")
    parser.add_argument("--n_ori", type=int, default=6, help="Number of orientations for the ICGNet planner.")
    parser.add_argument("--logdir", type=str, default="logs")

    parser.add_argument(
        "--config",
        type=str,
        default="/home/rene/ICRA_2024/icg_benchmark/data/51--0.656/config.yaml",
        help="Path to the config file of the ICGNet model.",
    )
    args = parser.parse_args()
    main(args)
