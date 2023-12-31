from ranger21 import Ranger21
import albumentations
from albumentations.pytorch import ToTensorV2
import torch
import math
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import ttach
from cfg import cfg
from resNet18 import ResNet18
from resNet50 import ResNet50
from resNet152 import ResNet152
import dataloader

def get_train_transforms():
    return albumentations.Compose([
        albumentations.CenterCrop(cfg.image_size, cfg.image_size, p = 1.0),
        albumentations.HorizontalFlip(p = 0.5),
        albumentations.VerticalFlip(p = 0.5),
        albumentations.Transpose(p = 0.5),
        albumentations.Rotate(limit = (-180, 180), p = 0.5),
        #albumentations.ShiftScaleRotate(shift_limit = 0.065, scale_limit = 0, rotate_limit = 180),
        albumentations.GaussianBlur(always_apply = False, blur_limit = 3, p = 0.2),
        #albumentations.Sharpen(alpha = (0.2, 0.5), lightness = (0.5, 1.0), always_apply = False, p = 0.2),
        #albumentations.GaussNoise(var_limit = (30.0, 30.0), mean = 0, always_apply = False, p = 0.2),
        #albumentations.CLAHE(clip_limit = 4.0, tile_grid_size=(4, 4), p = 0.3),
        albumentations.Cutout(max_h_size = int(cfg.image_size * 0.125), max_w_size = int(cfg.image_size * 0.125), num_holes = 1, p = 0.5),
        albumentations.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225], max_pixel_value = 255, p=1.0),
        ToTensorV2(p = 1.0)
    ])
def get_valid_transforms():
    return albumentations.Compose([
        albumentations.CenterCrop(height = cfg.image_size, width = cfg.image_size, p = 1.0),
        albumentations.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225], max_pixel_value = 255, p = 1.0),
        ToTensorV2(p=1.0)
    ])

def train_epoch(model, dataloader, loss_function, optimizer):
    model.train()
    total_loss = 0
    total_accuracy = 0
    with tqdm(dataloader, unit = 'Batch', desc = 'Train') as tqdm_loader:
        for index, (image_id, image, label) in enumerate(tqdm_loader):
            image = image.to(device = cfg.device)
            label = torch.tensor(label.to(device = cfg.device), dtype = torch.long)
            
            predict = model(image).to(device = cfg.device)
            loss = loss_function(predict, label)
            predict = predict.cpu().detach().argmax(dim = 1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # if index % cfg.optimizer_calculate:
            #     optimizer.step()
            #     optimizer.zero_grad()
            
            loss = loss.detach().item()
            total_loss = total_loss + loss
            accuracy = accuracy_score(predict, label.cpu())
            total_accuracy = total_accuracy + accuracy
            
            tqdm_loader.set_postfix(loss = loss, average_loss = total_loss/(index + 1), average_accuracy = total_accuracy/(index + 1))
    return total_accuracy/len(dataloader)
def valid_epoch(model, dataloader, loss_function, best_accuracy, best_loss):
    #model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        with tqdm(dataloader, unit = 'Batch', desc = 'Valid') as tqdm_loader:
            for index, (image_id, image, label) in enumerate(tqdm_loader):
                image = image.to(device = "cuda" if torch.cuda.is_available() else "cpu")
                label = torch.tensor(label.to(device = cfg.device), dtype = torch.long)
            
                predict = model(image).to(device = cfg.device)
                loss = loss_function(predict, label)
                predict = predict.cpu().detach().argmax(dim = 1)
            
                loss = loss.detach().item()
                total_loss = total_loss + loss
                accuracy = accuracy_score(predict, label.cpu())
                total_accuracy = total_accuracy + accuracy
            
                tqdm_loader.set_postfix(loss = loss, average_loss = total_loss/(index + 1), average_accuracy = total_accuracy/(index + 1))
        
        average_loss = total_loss/len(tqdm_loader)
        average_accuracy = total_accuracy/len(tqdm_loader)
        
        if average_accuracy > best_accuracy:
            print('best model update')
            best_accuracy = average_accuracy
            best_loss = average_loss
            torch.save(model,'model.pkl')
        elif average_accuracy == best_accuracy:
            if average_loss <= best_loss:
                print('best model update')
                best_loss = average_loss
                torch.save(model,'model.pkl')
    return average_accuracy, best_accuracy, best_loss
def train_and_valid(model, train_data_loader, valid_data_loader, loss_function, optimizer):
    train_accuracy_curve
    valid_accuarcy_curve
    best_accuracy = 0
    best_loss = 0
    for epoch in range(cfg.epoch):
        print('\nEpoch {}'.format(epoch + 1))
        
        train_accuracy = train_epoch(model, train_data_loader, loss_function, optimizer)
        tta_model = ttach.ClassificationTTAWrapper(model, ttach.aliases.d4_transform(), merge_mode='mean')
        valid_accuarcy, best_accuracy, best_loss = valid_epoch(tta_model, valid_data_loader, loss_function, best_accuracy, best_loss)
        
        train_accuracy_curve.append(train_accuracy)
        valid_accuarcy_curve.append(valid_accuarcy)
    return train_accuracy_curve, valid_accuarcy_curve, best_accuracy
def cross_validation(all_data_df, k = cfg.kfold):
    total_len = len(all_data_df)
    fold_len = int(total_len/k)
    dataloader_list = {'train': [], 'valid': []}
    for i in range(k):
        train_left_left_indices = 0
        train_left_right_indices = i * fold_len
        valid_left_indices = train_left_right_indices
        valid_right_indices = valid_left_indices + fold_len
        train_right_left_indices = valid_right_indices
        train_right_right_indices = total_len
        train_left_indices = list(range(train_left_left_indices, train_left_right_indices))
        train_right_indices = list(range(train_right_left_indices, train_right_right_indices))
        train_indices = train_left_indices + train_right_indices
        
        train_df = all_data_df.iloc[train_indices]
        valid_df = all_data_df.iloc[valid_left_indices : valid_right_indices]

        train_data_set = dataloader.LeukemiaDataset(train_df, transforms = get_train_transforms())
        valid_data_set = dataloader.LeukemiaDataset(valid_df, transforms = get_valid_transforms())

        train_data_loader = DataLoader(train_data_set, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)
        valid_data_loader = DataLoader(valid_data_set, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)

        dataloader_list['train'].append(train_data_loader)
        dataloader_list['valid'].append(valid_data_loader)

    accuracy_list = []

    for i in range(k):
        print('\nFold {}'.format(i + 1))
        model = ResNet18().to(cfg.device)
        cross_entropy = torch.nn.CrossEntropyLoss()
        ranger21 = Ranger21(model.parameters(), lr = cfg.lr, num_epochs = cfg.epoch, num_batches_per_epoch = len(dataloader_list['train'][i]))
        _, _, i_best_accuracy = train_and_valid(model, train_data_loader = dataloader_list['train'][i], valid_data_loader = dataloader_list['valid'][i], loss_function = cross_entropy, optimizer = ranger21, fold = i)
        accuracy_list.append(i_best_accuracy)
        model = torch.load('model.pkl').to(cfg.device)
        torch.save(model,'model_' + str(i + 1) + '.pkl')
    for i in range(k):
         print('\nFold {} best accuracy {}'.format(i + 1, accuracy_list[i]))
def train_and_valid_all_model_and_show_curve(train_data_loader, valid_data_loader, loss_function):
    accuracy_curve = {'ResNet18_train': [], 'ResNet18_valid': [], 'ResNet50_train': [], 'ResNet50_valid': [], 'ResNet152_train': [], 'ResNet152_valid': []}
    resnet18 = ResNet18().to(cfg.device)
    ranger21 = Ranger21(resnet18.parameters(), lr = cfg.lr, num_epochs = cfg.epoch, num_batches_per_epoch = len(train_data_loader))
    accuracy_curve['ResNet18_train'], accuracy_curve['ResNet18_valid'], _ = train_and_valid(model = resnet18, train_data_loader = train_data_loader, valid_data_loader = valid_data_loader, loss_function = loss_function, optimizer = ranger21)

    resnet50 = ResNet50().to(cfg.device)
    ranger21 = Ranger21(resnet50.parameters(), lr = cfg.lr, num_epochs = cfg.epoch, num_batches_per_epoch = len(train_data_loader))
    accuracy_curve['ResNet50_train'], accuracy_curve['ResNet50_valid'], _ = train_and_valid(model = resnet50, train_data_loader = train_data_loader, valid_data_loader = valid_data_loader, loss_function = loss_function, optimizer = ranger21)

    resnet152 = ResNet152().to(cfg.device)
    ranger21 = Ranger21(resnet152.parameters(), lr = cfg.lr, num_epochs = cfg.epoch, num_batches_per_epoch = len(train_data_loader))
    accuracy_curve['ResNet18_train'], accuracy_curve['ResNet152_valid'], _ = train_and_valid(model = resnet152, train_data_loader = train_data_loader, valid_data_loader = valid_data_loader, loss_function = loss_function, optimizer = ranger21)

    dataloader.show_curve(accuracy_curve)
# all_data_set = dataloader.LeukemiaDataset(dataloader.all_data_df)
# kfold(all_data_set)

train_transforms = get_train_transforms()
valid_transforms = get_valid_transforms()

train_data_set = dataloader.LeukemiaDataset(dataloader.train_df, transforms = train_transforms)
valid_data_set = dataloader.LeukemiaDataset(dataloader.valid_df, transforms = valid_transforms)
train_data_loader = DataLoader(train_data_set, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)
valid_data_loader = DataLoader(valid_data_set, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)

cross_entropy = torch.nn.CrossEntropyLoss()
resnet18 = ResNet18().to(cfg.device)
ranger21 = Ranger21(resnet18.parameters(), lr = cfg.lr, num_epochs = cfg.epoch, num_batches_per_epoch = len(train_data_loader))
train_and_valid(model = resnet18, train_data_loader = train_data_loader, valid_data_loader = valid_data_loader, loss_function = cross_entropy, optimizer = ranger21)

# train_and_valid_all_model_and_show_curve(train_data_loader, valid_data_loader, cross_entropy)