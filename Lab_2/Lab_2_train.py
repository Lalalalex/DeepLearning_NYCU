import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import augmentation
import dataloader

class cfg:
    seed = 'https://github.com'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    activate = 'ReLU'
    lr = 5e-3
    min_lr = 5e-8
    T0 = 40
    epoch = 200

def setSeed(seed = cfg.seed):
    seed = math.prod([ord(i) for i in seed])%(2**32)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def split_data(data, split = cfg.split):
    train, valid = {}, {}
    train['input'], valid['input'], train['label'], valid['label'] = train_test_split(data['input'], data['label'], train_size = split)
    return train, valid

class EEGDataSet:
    def __init__(self, data, is_test_model = False):
        self.data = data
        self.is_test_model = is_test_model
    def __getitem__(self, index):
        input = self.data['input'][index]
        if not self.is_test_model:
            input = augmentation.time_shift(input)
            input = augmentation.gaussian_noise(input)
        label = self.data['label'][index]
        return input, label
    def __len__(self):
        return self.data['input'].shape[0]
class EEGNet(nn.Module):
    def __init__(self, out_feature = 2, activate = cfg.activate):
        super().__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (1, 51), stride = (1, 1), padding = (0, 25), bias = False),
            nn.BatchNorm2d(16, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        )
        self.depthWiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (2, 1), stride = (1, 1), groups = 16, bias = False),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            self.activate_function(activate),
            nn.AvgPool2d(kernel_size = (1, 4), stride = (1, 4), padding = 0),
            nn.Dropout2d(p = 0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = (1, 15), stride = (1, 1), padding = (0, 7), bias = False),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            self.activate_function(activate),
            nn.AvgPool2d(kernel_size = (1, 8), stride = (1, 8), padding = 0),
            nn.Dropout2d(p = 0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 736, out_features = out_feature, bias = True)
        )
    def activate_function(self, activate):
        activate_functions = {
            'ELU': nn.ELU(alpha = 1),
            'ReLU': nn.ReLU(),
            'Leaky ReLU':nn.LeakyReLU(negative_slope = 0.05)
        }
        if activate in activate_functions:
            return activate_functions[activate]
        else:
            print('Activation function not found.')
    def forward(self, input):
        output = self.firstConv(input)
        output = self.depthWiseConv(output)
        output = self.separableConv(output)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output
class DeepConvNet(nn.Module):
    def __init__(self, out_feature = 2, activate = cfg.activate):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size = (1, 5), padding = 'valid'),
            nn.Conv2d(25, 25, kernel_size = (2, 1), padding = 'valid'),
            nn.BatchNorm2d(25, eps = 1e-5, momentum = 0.1),
            self.activate_function(activate),
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout2d(p = 0.5),
            nn.Conv2d(25, 50, kernel_size = (1, 5), padding = 'valid'),
            nn.BatchNorm2d(50, eps = 1e-5, momentum = 0.1),
            self.activate_function(activate),
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout2d(p = 0.5),
            nn.Conv2d(50, 100, kernel_size = (1, 5), padding = 'valid'),
            nn.BatchNorm2d(100, eps = 1e-5, momentum = 0.1),
            self.activate_function(activate),
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout2d(p = 0.5),
            nn.Conv2d(100, 200, kernel_size = (1, 5), padding = 'valid'),
            nn.BatchNorm2d(200, eps = 1e-5, momentum = 0.1),
            self.activate_function(activate),
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout2d(p = 0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 8600, out_features = out_feature, bias = True)
        )
    def activate_function(self, activate):
        activate_functions = {
            'ELU': nn.ELU(alpha = 1),
            'ReLU': nn.ReLU(),
            'Leaky ReLU':nn.LeakyReLU(negative_slope = 0.05)
        }
        if activate in activate_functions:
            return activate_functions[activate]
        else:
            print('Activation function not found.')
    def forward(self, input):
        output = self.network(input)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

def train_one_epoch(model, dataloader, loss_function, optimizer):
    model.train()
    total_loss = 0
    total_accuracy = 0
    with tqdm(dataloader, unit = 'batch', desc = 'Train') as tqdm_loader:
        for index, (input, label) in enumerate(tqdm_loader):
            input = input.to(cfg.device)
            input = torch.tensor(input, dtype = torch.float)
            label = label.to(cfg.device)
            label = torch.tensor(label, dtype = torch.long)

            predict = model(input)
            loss = loss_function(predict, label)
            predict = predict.cpu().detach().argmax(dim = 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.detach().item()
            total_loss = total_loss + current_loss
            current_accuracy = accuracy_score(predict, label.cpu())
            total_accuracy = total_accuracy + current_accuracy

            tqdm_loader.set_postfix(loss = current_loss ,avg_loss = total_loss/(index + 1), avg_accuracy = f'{total_accuracy/(index + 1):.4f}')
def eval_one_epoch(model, dataloader, loss_function):
    model.eval()
    total_loss = 0
    best_loss = 10
    total_accuracy = 0
    best_accuracy = 0
    with torch.no_grad():
        with tqdm(dataloader, unit = 'batch', desc='Valid') as tqdm_loader:
            for index, (input, label) in enumerate(tqdm_loader):
                input = input.to(cfg.device)
                input = torch.tensor(input, dtype = torch.float)
                label = label.to(cfg.device)
                label = torch.tensor(label, dtype=torch.long)

                predict = model(input)
                loss = loss_function(predict, label)
                predict = predict.cpu().detach().argmax(dim = 1)

                current_loss = loss.detach().item()
                total_loss = total_loss + current_loss
                current_accuracy = accuracy_score(predict, label.cpu())
                total_accuracy = total_accuracy + current_accuracy

                tqdm_loader.set_postfix(loss = current_loss ,avg_loss = total_loss/(index + 1), avg_accuracy = f'{total_accuracy/(index + 1):.4f}')
            avg_loss = total_loss/len(tqdm_loader)
            avg_accuracy = total_accuracy/len(tqdm_loader)
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_loss = avg_loss
                torch.save(model.state_dict(),'best_model.pkl')
            elif avg_accuracy == best_accuracy:
                if avg_loss <= best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(),'best_model.pkl') 
def test(model, data_loader):
    with torch.no_grad():
        with tqdm(data_loader, unit = 'Iter', desc='Test') as tqdm_loader:
        
            for index, (input, label) in enumerate(tqdm_loader):
                input = input.to(cfg.device)
                input = torch.tensor(input, dtype = torch.float)
                label = label.to(cfg.device)
                label = torch.tensor(label, dtype=torch.long)

                predict = model(input)
                predict = predict.cpu().detach().argmax(dim = 1)

                accuracy = accuracy_score(predict, label.cpu())

                tqdm_loader.set_postfix(accuracy = f'{accuracy:.4f}')

setSeed(cfg.seed)

train_data = {}
test_data = {}
train_data['input'], train_data['label'], test_data['input'], test_data['label'] = dataloader.read_bci_data()

train_data, valid_data = split_data(train_data)

train_data_set = EEGDataSet(train_data)
valid_data_set = EEGDataSet(valid_data, is_test_model = True)
test_data_set = EEGDataSet(test_data, is_test_model = True)

train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size = cfg.batch_size, pin_memory = True, drop_last = False, num_workers = 4)
valid_data_loader = torch.utils.data.DataLoader(valid_data_set, batch_size = cfg.batch_size, pin_memory = True, drop_last = False, num_workers = 4)
test_data_loader = torch.utils.data.DataLoader(test_data_set, batch_size = cfg.batch_size, pin_memory = True, drop_last = False, num_workers = 4)

model = DeepConvNet().to(cfg.device)

cross_entropy = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch = -1)

for epoch in range(cfg.epoch):
    print('\nEpoch {}'.format(epoch))
    train_one_epoch(model, train_data_loader, cross_entropy, optimizer)
    eval_one_epoch(model, valid_data_loader, cross_entropy)
    scheduler.step()

model.load_state_dict(torch.load('best_model.pkl'))

test(model, test_data_loader)