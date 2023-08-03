import os
import torch
import math
import numpy as np
import pandas as pd
import random

class cfg:
    seed = 'Leaderboard'
    image_size  = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epoch = 100
    optimizer_calculate = 1
    kfold = 5
    lr = 2e-2
    split = 0.85

def setSeed(seed = cfg.seed):
    seed = math.prod([ord(i) for i in cfg.seed])%(2**32)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pd.core.common.random_state(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
setSeed()