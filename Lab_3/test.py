import ttach as tta
import pandas as pd
import numpy as np
import math
import os
import random
from collections import Counter
import cv2
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import timm
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import albumentations
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize)
from albumentations.pytorch import ToTensorV2
import dataset
from cfg import cfg

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

def get_test_transforms():
    return albumentations.Compose([
        albumentations.CenterCrop(p=1.0, height = cfg.image_size, width = cfg.image_size),
        albumentations.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], max_pixel_value = 255, p = 1.0),
        ToTensorV2(p=1.0)
    ])


def test(model,dataloader):
    model.eval()
    pred_list =[]
    pred_id_list =[]
    with torch.no_grad():
        with tqdm(dataloader,unit='batch',desc='Test') as tqdm_loader:
            for idx, (imgid, img) in enumerate(tqdm_loader):

                img = img.to(device=cfg.device)
                pred = model(img).detach().cpu().argmax(dim=1)

                pred_list.append(pred)
                pred_id_list.append(imgid)
    pred_list = np.concatenate(pred_list,axis=0)
    pred_id_list =  np.concatenate(pred_id_list,axis=0)
    return pred_list,pred_id_list

model = torch.load('model.pkl').to(cfg.device)
test_transforms = get_test_transforms()
test_data_set = dataset.LeukemiaDataset(dataset.test_df, transforms = test_transforms, is_test_model = True)
test_data_loader = DataLoader(test_data_set, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)

model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
pred_list,pred_id_list = test(model, test_data_loader)

submit_df  = pd.DataFrame()
submit_df['ID'] = pred_id_list
submit_df['label']=pred_list
submit_df.to_csv(os.path.join('./', 'asdsubmission.csv'),index=False)