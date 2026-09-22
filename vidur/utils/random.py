import os
import random

import numpy as np


def set_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch is optional at import time for non-RL runs; seed it when available so
    # Q-network initialisation (and therefore RL runs) is reproducible.
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
