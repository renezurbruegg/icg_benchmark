import json
import os
from pathlib import Path

import numpy as np
import torch
import tqdm
from icg_net.wrapper.model_wrapper import ModelWrapper
from torch.utils.data.dataloader import default_collate

from icg_benchmark.third_party.ConvOnet.dataset import DatasetVoxelOccGeo
from icg_benchmark.third_party.ConvOnet.generation import Generator3D
from icg_benchmark.third_party.ConvOnet.mesh_evaluator import MeshEvaluator


def evaluate_network(
    model: ModelWrapper,
    main_folder: Path,
    root: Path,
    raw_root: Path,
    threshold: float = 0.5,
    resolution: int = 16,
    show: bool = False,
    with_table=False,
    device="cuda",
):

    mean_dict = {
        "iou": [],
        "chamfer-L1": [],
        "normals accuracy": [],
        "chamfer-L2": [],
        "f-score": [],
        "completeness": [],
        "accuracy": [],
        "completeness2": [],
        "accuracy2": [],
        "f-score-15": [],  # threshold = 1.5%
        "f-score-20": [],  # threshold = 2.0%
    }

    # create data loaders
    test_set, test_loader, size = create_test_loader(root, raw_root, with_table=with_table)

    generator = Generator3D(
        model,
        resolution0=resolution,
        upsampling_steps=3,
        device=device,
        threshold=threshold,
        input_type="pointcloud",
        padding=0,
    )

    logdir = main_folder / "reconstruction"
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging reconstruction to {logdir}")

    with torch.no_grad():
        for idx, (data, gt_mesh) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True):

            pc_in, points_occ, occ = data
            pc_in = pc_in.float().to(device)
            points_occ = points_occ.float().to(device)
            occ = occ.float().to(device)
            gt_mesh = gt_mesh[0]
            # gt_mesh.vertices = gt_mesh.vertices / test_set.size - 0.5

            pred_mesh = predict_mesh(generator, pc_in)
            pred_mesh.vertices = (pred_mesh.vertices + 0.5) * test_set.size
            points_occ = (points_occ + 0.5) * test_set.size

            # if not test_set.normalize:
            #     points_occ = points_occ/test_set.size - 0.5
            if show:
                pred_mesh.show()
            # import open3d as o3d
            # pc = o3d.geometry.PointCloud()
            # pc.points= o3d.utility.Vector3dVector(pc_in.squeeze().cpu().numpy())
            # o3d.visualization.draw(pc)

            pred_mesh.export(logdir / f"scene_{idx}.obj")
            # trimesh.smoothing.filter_laplacian(pred_mesh, iterations=5)
            # # smooth it
            # pred_mesh.export(logdir / f"scene_{idx}_smooth.obj")
            gt_mesh.export(logdir / f"scene_{idx}_gt.obj")

            out_dict = eval_mesh(pred_mesh, gt_mesh, points_occ, occ)
            save_dir = logdir / ("%05d" % idx)
            os.makedirs(str(save_dir), exist_ok=True)

            if not "empty" in out_dict.keys():
                for k, v in mean_dict.items():
                    if isinstance(v, list):
                        if out_dict[k] >= -1e5:  # avoid nan
                            mean_dict[k].append(out_dict[k])
                            print(k, out_dict[k])
            else:
                print(f"{idx} empty mesh!")
            with open(save_dir / "results.json", "w") as f:
                json.dump({k: float(v) for k, v in out_dict.items()}, f, indent=4)

    print("Geometry prediction results:")
    out_dict = {}
    for k, v in mean_dict.items():
        if isinstance(v, list):
            print("%s: %.6f" % (k, np.mean(v)))
            out_dict[k] = float(np.mean(v))
            out_dict[k + "_std"] = float(np.std(v))
    with open(logdir / "mean_results.json", "w") as f:
        json.dump({k: v for k, v in out_dict.items()}, f, indent=4)


def create_test_loader(root, root_raw, num_point_occ=100000, with_table=True):
    # load the dataset
    def collate_fn(batch):
        meshes = [d[-1] for d in batch]
        batch = [(d[0], d[1], d[2]) for d in batch]
        return default_collate(batch), meshes

    dataset = DatasetVoxelOccGeo(root, root_raw, num_point_occ=num_point_occ)

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn
    )

    return dataset, test_loader, dataset.size


def predict_mesh(generator, pc_input):
    pred_mesh, _ = generator.generate_mesh({"inputs": pc_input})
    return pred_mesh


def eval_mesh(pred_mesh, gt_mesh, points_occ, occ):
    evaluator = MeshEvaluator()
    pointcloud_tgt, idx_tgt = gt_mesh.sample(evaluator.n_points, True)
    normals_tgt = gt_mesh.face_normals[idx_tgt]
    points_occ = points_occ[0].cpu().numpy()
    occ = occ[0].cpu().numpy()
    out = evaluator.eval_mesh(pred_mesh, pointcloud_tgt.astype(np.float32), normals_tgt, points_occ, occ)
    return out
