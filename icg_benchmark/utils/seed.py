import numpy as np
import torch
import torch.backends.cudnn as cudnn


def seed_everything(seed=1234):
    np.random.seed(seed)
    torch.set_num_threads(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
