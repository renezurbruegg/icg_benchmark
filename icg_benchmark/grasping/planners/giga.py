from .base import GraspPlannerModule
from icg_benchmark.grasping.preprocessing.tsdf import GigaObservation
from icg_benchmark.third_party.ConvOnet.detection_implicit import VGNImplicit


class GigaPlanner(GraspPlannerModule[GigaObservation]):

    def __init__(self, model: VGNImplicit, device="cuda") -> None:
        super().__init__()
        self.device = device
        self.model = model

    def forward(self, tsdf_cb, timer):
        with timer["preprocess"]:
            state = tsdf_cb(timer)
        with timer["predict"]:
            grasps = self.model(state=state, timer=timer)
        return grasps
