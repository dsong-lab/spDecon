"""Explicit RNG and deterministic CUDA settings for repeatable new analyses."""
import os
import random
import numpy as np
import torch


def set_seed(seed=0, *, deterministic=False):
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
