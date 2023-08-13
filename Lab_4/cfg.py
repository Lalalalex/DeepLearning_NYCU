import warnings
warnings.filterwarnings('ignore')
import os
import torch
import math
import numpy as np
import pandas as pd
import random
import requests

class cfg:
    seed = 'Lab 4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setSeed(seed = cfg.seed):
    seed = math.prod([ord(i) for i in cfg.seed])%(2**32)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pd.core.common.random_state(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
def send_message(message = ''):
    headers = {"Authorization": "Bearer " + 'WymDnN4bjKRdD8gkluCKklAoizkDSRaxKiJSQnKdgAO'}
    data = { 'message': message }
    requests.post("https://notify-api.line.me/api/notify", headers = headers, data = data)
