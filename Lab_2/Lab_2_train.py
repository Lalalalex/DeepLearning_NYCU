import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from ranger21 import Ranger21
import augmentation
import dataloader

class cfg:
    seed = 'Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    model = 'EEGNet'
    activate = 'ReLU'
    optimizer = 'AdamW'
    loss_function = 'cross_entropy'
    lr = 5e-2
    epoch = 300

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
def show_curve(data, file_name = 'learning_curve', title = 'Learning Curve', x = 'epoch', y = 'accuracy'):
    plt.figure()
    plt.title(title)
    for i in data:
        plt.plot(data[i], label = i)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.savefig(file_name)

def train_transform(data):
    data = augmentation.time_shift(data, p = 0.7, max_shift = 50)
    data = augmentation.gaussian_noise(data, p = 0.2, limit = 2.2)
    return data

class EEGDataSet(Dataset):
    def __init__(self, data, transform = None, is_test_model = False):
        self.data = data
        self.is_test_model = is_test_model
        self.transform = transform
    def __getitem__(self, index):
        input = self.data['input'][index]
        if not self.is_test_model:
            input = self.transform(input)
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
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size = (1, 5), padding = 'valid'),
            nn.Conv2d(25, 25, kernel_size = (2, 1), padding = 'valid'),
            nn.BatchNorm2d(25, eps = 1e-5, momentum = 0.1),
            self.activate_function(activate),
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout2d(p = 0.5)
        )
        self.secondConv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size = (1, 5), padding = 'valid'),
            nn.BatchNorm2d(50, eps = 1e-5, momentum = 0.1),
            self.activate_function(activate),
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout2d(p = 0.5)
        )
        self.thirdConv = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size = (1, 5), padding = 'valid'),
            nn.BatchNorm2d(100, eps = 1e-5, momentum = 0.1),
            self.activate_function(activate),
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout2d(p = 0.5)
        )
        self.forthConv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size = (1, 5), padding = 'valid'),
            nn.BatchNorm2d(200, eps = 1e-5, momentum = 0.1),
            self.activate_function(activate),
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout2d(p = 0.5)
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
        output = self.firstConv(input)
        output = self.secondConv(output)
        output = self.thirdConv(output)
        output = self.forthConv(output)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

def train_one_epoch(model, dataloader, loss_function, optimizer):
    model.train()
    total_loss = 0
    total_accuracy = 0
    avg_accuracy = 0
    for index, (input, label) in enumerate(dataloader):
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
        avg_accuracy = total_accuracy/(index + 1)
    return avg_accuracy
def test_one_epoch(model, test_data, loss_function, best_accuracy, best_loss):
    model.eval()
    with torch.no_grad():
        input = test_data['input']
        input = torch.from_numpy(input).to(cfg.device)
        input = torch.tensor(input, dtype = torch.float)

        label = test_data['label']
        label = torch.from_numpy(label).to(cfg.device)
        label = torch.tensor(label, dtype = torch.long)

        predict = model(input)
        loss = loss_function(predict, label)
        predict = predict
        predict = predict.cpu().detach().argmax(dim = 1)

        accuracy = accuracy_score(predict, label.cpu())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_loss = loss
            torch.save(model.state_dict(),'best_model.pkl')
        elif accuracy == best_accuracy:
            if loss <= best_loss:
                best_loss = loss
                torch.save(model.state_dict(),'best_model.pkl')
    return accuracy, best_accuracy, best_loss
def train_and_test(title, model = cfg.model, activate = cfg.activate, optimizer = cfg.optimizer, loss_function = cfg.loss_function, batch_size = cfg.batch_size):
    train_accuracy_curve = []
    test_accuracy_curve = []
    setSeed(cfg.seed)
    train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size = batch_size, pin_memory = True, drop_last = False, num_workers = 4)
    models = {
        'EEGNet': EEGNet(activate = activate),
        'DeepConvNet': DeepConvNet(activate = activate)
    }
    current_model =  models[model].to(cfg.device)
    optimizers = {
        'Ranger21': Ranger21(current_model.parameters(), lr = cfg.lr, num_epochs = cfg.epoch, num_batches_per_epoch = len(train_data_loader)),
        'AdamW': torch.optim.AdamW(current_model.parameters(), lr = cfg.lr),
        'SGD': torch.optim.SGD(current_model.parameters(), lr = cfg.lr, momentum = 0.9, nesterov = True)
    }
    current_optimizer = optimizers[optimizer]
    loss_functions = {
        'cross_entropy': torch.nn.CrossEntropyLoss()
    }
    current_loss_function = loss_functions[loss_function]

    best_loss = 10
    best_accuracy = -10
    
    with tqdm(range(cfg.epoch), desc = title) as tqdm_loader:
        for epoch in tqdm_loader:
            train_accuracy = train_one_epoch(current_model, train_data_loader, current_loss_function, current_optimizer)
            test_accuracy, best_accuracy, best_loss = test_one_epoch(current_model, test_data, current_loss_function, best_accuracy, best_loss)
            train_accuracy_curve.append(train_accuracy)
            test_accuracy_curve.append(test_accuracy)
            tqdm_loader.set_postfix(train_accuracy = f'{train_accuracy:.4f}', test_accuracy = f'{test_accuracy:.4f}', best_accuracy = f'{best_accuracy:.4f}')
    return train_accuracy_curve, test_accuracy_curve

setSeed(cfg.seed)
train_data = {}
test_data = {}
train_data['input'], train_data['label'], test_data['input'], test_data['label'] = dataloader.read_bci_data()

train_data, valid_data = split_data(train_data)
train_data_set = EEGDataSet(train_data, transform = train_transform)

model_list = ['EEGNet', 'DeepConvNet']
activate_list = ['ReLU', 'Leaky ReLU', 'ELU']
batch_size_list = [32, 64, 128, 256, 512]

for model in model_list:
    accuracy = {}
    for activate in activate_list:
        train_accuracy_curve, test_accuracy_curve = train_and_test(title = model + ' ' + activate, model = model, activate = activate)
        accuracy[activate + '_train'] = train_accuracy_curve
        accuracy[activate + '_test'] = test_accuracy_curve
    show_curve(accuracy, file_name = model + '_activate')
    accuracy = {}
    for batch_size in batch_size_list:
        train_accuracy_curve, test_accuracy_curve = train_and_test(title = model + ' ' + str(batch_size), model = model, batch_size = batch_size)
        accuracy[str(batch_size) + '_train'] = train_accuracy_curve
        accuracy[str(batch_size) + '_test'] = test_accuracy_curve
    show_curve(accuracy, file_name = model + '_batch_size')