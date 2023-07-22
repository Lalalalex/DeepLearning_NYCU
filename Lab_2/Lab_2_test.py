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

base_data_path = '/home/pp037/DeepLearning_NYCU/Lab_2/data'
base_model_path = '/home/pp037/DeepLearning_NYCU/Lab_2/model'

class cfg:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activate = 'ReLU'
    model_path = os.path.join(base_model_path, 'EEGNet_ReLU_0.8722.pkl')

test_data = {}
junk, junk, test_data['input'], test_data['label'] = dataloader.read_bci_data()

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
        print('accuracy = {:.4f}'.format(accuracy))

model = EEGNet().to(cfg.device)
model.load_state_dict(torch.load(cfg.model_path))
test(model, test_data)