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
import dataloader
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

model_1 = torch.load('model_1.pkl').to(cfg.device)
model_1 = tta.ClassificationTTAWrapper(model_1, tta.aliases.d4_transform(), merge_mode='mean')
model_2 = torch.load('model_2.pkl').to(cfg.device)
model_2 = tta.ClassificationTTAWrapper(model_2, tta.aliases.d4_transform(), merge_mode='mean')
model_3 = torch.load('model_3.pkl').to(cfg.device)
model_3 = tta.ClassificationTTAWrapper(model_3, tta.aliases.d4_transform(), merge_mode='mean')
model_4 = torch.load('model_4.pkl').to(cfg.device)
model_4 = tta.ClassificationTTAWrapper(model_4, tta.aliases.d4_transform(), merge_mode='mean')
model_5 = torch.load('model_5.pkl').to(cfg.device)
model_5 = tta.ClassificationTTAWrapper(model_5, tta.aliases.d4_transform(), merge_mode='mean')
model_6 = torch.load('model_5.pkl').to(cfg.device)
model_6 = tta.ClassificationTTAWrapper(model_5, tta.aliases.d4_transform(), merge_mode='mean')
model_7 = torch.load('model_5.pkl').to(cfg.device)
model_7 = tta.ClassificationTTAWrapper(model_5, tta.aliases.d4_transform(), merge_mode='mean')
test_transforms = get_test_transforms()

test_data_set = dataloader.LeukemiaDataset(dataloader.test_df, transforms = test_transforms, is_test_model = True)
test_data_loader = DataLoader(test_data_set, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)

def test_addition(model_1, model_2, model_3, model_4, model_5, model_6, model_7,  dataloader):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()
    model_6.eval()
    model_7.eval()
    pred_list =[]
    pred_id_list =[]
    with torch.no_grad():
        with tqdm(dataloader,unit = 'batch',desc = 'Test') as tqdm_loader:
            for idx, (imgid, img) in enumerate(tqdm_loader):
                img = img.to(device=cfg.device)
                pred_1 = model_1(img).detach().cpu()
                pred_2 = model_2(img).detach().cpu()
                pred_3 = model_3(img).detach().cpu()
                pred_4 = model_4(img).detach().cpu()
                pred_5 = model_5(img).detach().cpu()
                pred_6 = model_4(img).detach().cpu()
                pred_7 = model_5(img).detach().cpu()

                pred = pred_1 + pred_2 + pred_3 + pred_4 + pred_5 + pred_6 + pred_7

                pred = pred.argmax(dim = 1)

                pred_list.append(pred)
                pred_id_list.append(imgid)
    pred_list = np.concatenate(pred_list,axis = 0)
    pred_id_list =  np.concatenate(pred_id_list,axis = 0)
    return pred_list, pred_id_list

def test_vote(model_1, model_2, model_3, model_4, model_5,model_6, model_7, dataloader):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()
    model_6.eval()
    model_7.eval()
    pred_list =[]
    pred_id_list =[]
    with torch.no_grad():
        with tqdm(dataloader,unit = 'batch',desc = 'Test') as tqdm_loader:
            for idx, (imgid, img) in enumerate(tqdm_loader):
                img = img.to(device=cfg.device)
                pred_1 = model_1(img).detach().cpu().argmax(dim = 1)
                pred_2 = model_2(img).detach().cpu().argmax(dim = 1)
                pred_3 = model_3(img).detach().cpu().argmax(dim = 1)
                pred_4 = model_4(img).detach().cpu().argmax(dim = 1)
                pred_5 = model_5(img).detach().cpu().argmax(dim = 1)
                pred_6 = model_4(img).detach().cpu().argmax(dim = 1)
                pred_7 = model_5(img).detach().cpu().argmax(dim = 1)

                pred = pred_1 + pred_2 + pred_3 + pred_4 + pred_5 + pred_6 + pred_7

                for index, pred_sum in enumerate(pred):
                    if pred_sum >= 4:
                        pred[index] = 1
                    else:
                        pred[index] = 0

                pred_list.append(pred)
                pred_id_list.append(imgid)
    pred_list = np.concatenate(pred_list,axis = 0)
    pred_id_list =  np.concatenate(pred_id_list,axis = 0)
    return pred_list, pred_id_list

pred_list, pred_id_list = test_vote(model_1, model_2, model_3, model_4, model_5, model_6, model_7, test_data_loader)

submit_df  = pd.DataFrame()
submit_df['ID'] = pred_id_list
submit_df['label'] = pred_list
submit_df.to_csv(os.path.join('./', 'submission.csv'),index = False)