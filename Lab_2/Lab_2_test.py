import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import dataloader
from torchsummary import summary


base_data_path = '/home/pp037/DeepLearning_NYCU/Lab_2/data'
base_model_path = '/home/pp037/DeepLearning_NYCU/Lab_2/model'

class cfg:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activate = 'ELU'
    model_path = os.path.join(base_model_path, 'EEGNet_ELU_0.8546.pkl')

test_data = {}
junk, junk, test_data['input'], test_data['label'] = dataloader.read_bci_data()

class EEGNet(nn.Module):
    def __init__(self, out_feature = 2, activate = cfg.activate):
        super().__init__()
        self.activate = activate
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
    def get_name(self):
        return 'EEGNet'
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
        self.activate = activate
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
            'Leaky ReLU':nn.LeakyReLU(),
            'SiLU': nn.SiLU()
        }
        if activate in activate_functions:
            return activate_functions[activate]
        else:
            print('Activation function not found.')
    def get_name(self):
        return 'DeepConvNet'
    def forward(self, input):
        output = self.firstConv(input)
        output = self.secondConv(output)
        output = self.thirdConv(output)
        output = self.forthConv(output)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output
class MyNet(nn.Module):
    def __init__(self, out_feature = 2, activate = cfg.activate):
        super().__init__()
        self.activate = activate
        self.time_filter = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = (1, 51), stride = (1, 1), padding = (0, 25), bias = False),
            nn.BatchNorm2d(8, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        )
        self.depthConv_1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size = (2, 1), stride = (1, 1), bias = False),
            nn.BatchNorm2d(16, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            self.activate_function(activate),
            nn.Conv2d(16, 16, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(16, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.Dropout2d(p = 0.3)
        )
        self.shortcut_1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size = (2, 1), stride = 1, bias = False),
            nn.BatchNorm2d(16, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        )
        self.depthConv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (1, 15), stride = (1, 1), padding = (0, 7), bias = False),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            self.activate_function(activate),
            nn.Conv2d(32, 32, kernel_size = (1, 15), stride = (1, 1), padding = (0, 7), bias = False),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.Dropout2d(p = 0.3)
        )
        self.shortcut_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        )
        self.shortcut_3 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size = (1, 25), stride = (1, 25), bias = False),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        )
        self.pooling = nn.AvgPool2d(kernel_size = (1, 25), stride = (1, 25), padding = 0)
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 64, out_features = out_feature, bias = True)
        )
    def activate_function(self, activate):
        activate_functions = {
            'ELU': nn.ELU(alpha = 1),
            'ReLU': nn.ReLU(),
            'Leaky ReLU': nn.LeakyReLU(),
            'SiLU': nn.SiLU()
        }
        if activate in activate_functions:
            return activate_functions[activate]
        else:
            print('Activation function not found.')
    def get_name(self):
        return 'MyNet'
    def forward(self, input):
        output = self.time_filter(input)
        tmp = self.shortcut_3(output)
        output = self.shortcut_1(output) + self.depthConv_1(output)
        output = self.pooling(output)
        output = self.shortcut_2(output) + self.depthConv_2(output) + tmp
        output = self.pooling(output)
        output = output.view(output.size(0), -1)    
        output = self.classifier(output)
        return output

def test(model, test_data):
    model.eval()
    with torch.no_grad():
        input = test_data['input']
        input = torch.from_numpy(input).to(cfg.device)
        input = torch.tensor(input, dtype = torch.float)

        label = test_data['label']
        label = torch.from_numpy(label).to(cfg.device)
        label = torch.tensor(label, dtype = torch.long)

        predict = model(input)
        predict = predict
        predict = predict.cpu().detach().argmax(dim = 1)

        accuracy = accuracy_score(predict, label.cpu())
        print(f'Model: {model.get_name():12} | Activation Function: {model.activate:15} | Accuracy: {accuracy:.4f}')
        #print(f'')



model = EEGNet(activate = 'ReLU').to(cfg.device)
model.load_state_dict(torch.load('model/EEGNet_ReLU_0.8722.pkl'))
test(model, test_data)

model = EEGNet(activate = 'Leaky ReLU').to(cfg.device)
model.load_state_dict(torch.load('model/EEGNet_Leaky_ReLU_0.8676.pkl'))
test(model, test_data)

model = EEGNet(activate = 'ELU').to(cfg.device)
model.load_state_dict(torch.load('model/EEGNet_ELU_0.8546.pkl'))
test(model, test_data)

model = DeepConvNet(activate = 'ReLU').to(cfg.device)
model.load_state_dict(torch.load('model/DeepConvNet_ReLU_0.8213.pkl'))
test(model, test_data)

model = DeepConvNet(activate = 'Leaky ReLU').to(cfg.device)
model.load_state_dict(torch.load('model/DeepConvNet_Leaky_ReLU_0.8130.pkl'))
test(model, test_data)

model = DeepConvNet(activate = 'ELU').to(cfg.device)
model.load_state_dict(torch.load('model/DeepConvNet_ELU_0.8065.pkl'))
test(model, test_data)

model = MyNet(activate = 'ReLU').to(cfg.device)
model.load_state_dict(torch.load('model/MyNet_0.8713.pkl'))
test(model, test_data)