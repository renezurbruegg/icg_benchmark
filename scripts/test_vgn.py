from icg_benchmark.grasping.eval import GraspEvaluator

from icg_benchmark.grasping.planners import EdgeGraspPlanner
from icg_benchmark.grasping.preprocessing import GigaTSDFObservation
from icg_benchmark.grasping.view_samplers import top_down_view

from icg_benchmark.third_party.ConvOnet.detection import VGN

import argparse
import datetime
import os

from icg_benchmark.models.edge_grasp.vn_edge_grasper import EdgeGrasper as VNEdgeGrasper
from icg_benchmark.models.edge_grasp.edge_grasper import EdgeGrasper



def main(args):

    planner = VGN(
        args.checkpoint,
        model_type="vgn",
        best=True,
        qual_th=args.th,
        force_detection=True,
        out_th=args.th
    )

    evaluator = GraspEvaluator(
        scene=args.scene,
        object_set=args.object_set,
        show_gui=args.sim_gui,
        rand=args.rand,
        preproc=GigaTSDFObservation,
    )

    log_file = args.name + "_" + args.scene + "_" + args.object_set.replace("/", "_")
    log_file += datetime.datetime.now().strftime("_%m_%d_%H_%M_%S") + ".json"

    evaluator.eval_method(
        planner, static_view_fnc=top_down_view if args.top_down else None, log_file=os.path.join(args.logdir, log_file)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument("--scene", type=str, choices=["pile", "packed", "obj", "egad"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim-gui", action="store_true", default=False)
    parser.add_argument(
        "--rand", action="store_true", default=True, help="Randomize the scene, by loading objects at random poses."
    )
    parser.add_argument("--th", type=float, default=0.85, help="Threshold for the ICGNet planner.")

    parser.add_argument("--name", type=str, default="edge_grasp")
    parser.add_argument("--method", type=str, default="edge", choices=["edge", "edge-vn"])
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--top-down", action="store_true", default=False)

    parser.add_argument("--checkpoint", type=str, default="data/models/vgn_packed.pt")

    args = parser.parse_args()
    main(args)
