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

class cfg:
    seed = 'Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    activate = 'ELU'
    lr = 1e-3
    min_lr = 5e-8
    T0 = 50
    epoch = 500

def setSeed(seed = cfg.seed):
    seed = math.prod([ord(i) for i in seed])%(2**32)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
setSeed(cfg.seed)

base_data_path = '/home/pp037/DeepLearning_NYCU/Lab_2/data'
train_data_path = os.path.join(base_data_path, 'train_data.pkl')
test_data_path = os.path.join(base_data_path, 'test_data.pkl')

with open(train_data_path, 'rb') as f:
    train_data = pickle.load(f)
with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)

def split_data(data, split = cfg.split):
    train, valid = {}, {}
    train['input'], valid['input'], train['label'], valid['label'] = train_test_split(data['input'], data['label'], train_size = split)
    return train, valid

train_data, valid_data = split_data(train_data)

class EEGDataSet:
    def __init__(self, data, is_test_model = False):
        self.data = data
        self.is_test_model = is_test_model
    def __getitem__(self, index):
        input = self.data['input'][index]
        label = self.data['label'][index]
        return input, label
    def __len__(self):
        return self.data['input'].shape[0]

class EEGModel(nn.Module):
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
            'Leaky ReLU':nn.LeakyReLU()
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

train_data_set = EEGDataSet(train_data)
valid_data_set = EEGDataSet(valid_data)
test_data_set = EEGDataSet(test_data)

train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size = cfg.batch_size, pin_memory = True, drop_last = False, num_workers = 4)
valid_data_loader = torch.utils.data.DataLoader(valid_data_set, batch_size = cfg.batch_size, pin_memory = True, drop_last = False, num_workers = 4)
test_data_loader = torch.utils.data.DataLoader(test_data_set, batch_size = cfg.batch_size, pin_memory = True, drop_last = False, num_workers = 4)

model = EEGModel().to(cfg.device)

cross_entropy = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch=-1)

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

            tqdm_loader.set_postfix(loss = current_loss ,avg_loss = total_loss/(index + 1), avg_accuracy = total_accuracy/(index + 1))

def eval_one_epoch(model, dataloader, loss_function):
    model.eval
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

                tqdm_loader.set_postfix(loss = current_loss ,avg_loss = total_loss/(index + 1), avg_accuracy = total_accuracy/(index + 1))
            avg_loss = total_loss/len(tqdm_loader)
            avg_accuracy = total_accuracy/len(tqdm_loader)
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_loss = avg_loss
                torch.save(model.state_dict(),'../best_model.pkl')
                torch.save(model.state_dict(),'./best_model.pkl') 
            elif avg_accuracy == best_accuracy:
                if avg_loss <= best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(),'../best_model.pkl') 
                    torch.save(model.state_dict(),'./best_model.pkl') 

for epoch in range(cfg.epoch):
    print('\nEpoch {}'.format(epoch))
    train_one_epoch(model, train_data_loader, cross_entropy, optimizer)
    eval_one_epoch(model, valid_data_loader, cross_entropy)
    scheduler.step()

model.load_state_dict(torch.load('./best_model.pkl'))

def test(model, data_loader):
    with tqdm(data_loader, unit = 'Iter', desc='Test') as tqdm_loader:
        for index, (input, label) in enumerate(tqdm_loader):
            input = input.to(cfg.device)
            input = torch.tensor(input, dtype = torch.float)
            label = label.to(cfg.device)
            label = torch.tensor(label, dtype=torch.long)

            predict = model(input)
            predict = predict.cpu().detach().argmax(dim = 1)

            accuracy = accuracy_score(predict, label.cpu())

            tqdm_loader.set_postfix(accuracy = accuracy)

test(model, test_data_loader)