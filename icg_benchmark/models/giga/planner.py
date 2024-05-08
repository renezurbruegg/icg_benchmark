from torch import nn

from icg_benchmark.third_party.ConvOnet.detection_implicit import VGNImplicit

# def __call__(self, state, scene_mesh=None, aff_kwargs={}, return_reconstruction=False):
#     if hasattr(state, 'tsdf_process'):
#         tsdf_process = state.tsdf_process
#     else:
#         tsdf_process = state.tsdf

#     if isinstance(state.tsdf, np.ndarray):
#         tsdf_vol = state.tsdf
#         voxel_size = 0.3 / self.resolution
#         size = 0.3
#     else:
#         tsdf_vol = state.tsdf.get_grid()
#         voxel_size = tsdf_process.voxel_size
#         tsdf_process = tsdf_process.get_grid()
#         size = state.tsdf.size

#     with timer['predict']:
#         tic = time.time()
#         qual_vol, rot_vol, width_vol, embeddings = predict(tsdf_vol, self.pos, self.net, self.device, coll_check = self.coll_check)

#     with timer['process']:
#         qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
#         rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
#         width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

#         qual_vol, rot_vol, width_vol = process(tsdf_process, qual_vol, rot_vol, width_vol, out_th=self.out_th)
#         qual_vol = bound(qual_vol, voxel_size)


class EdgeGraspPlanner(nn.Module):

    def __init__(self, model: VGNImplicit, device="cuda") -> None:
        super().__init__()
        self.device = device
        self.model = model

    def forward(self, tsdf_cb, timer):
        with timer["preprocess"]:
            state = tsdf_cb()
        with timer["predict"]:
            grasps = self.model(state=state)
        return grasps
