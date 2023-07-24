# EEGNet_ReLU_0.8722
- path:/home/pp037/DeepLearning_NYCU/Lab_2/model/EEGNet_ReLU_0.8722.pkl
## Configs
```python
class cfg:
    seed = 'Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    model = 'EEGNet'
    activate = 'ReLU'
    optimizer = 'Ranger21'
    loss_function = 'cross_entropy'
    lr = 5e-2
    epoch = 300
```
```python
def train_transform(data):
    data = augmentation.time_shift(data, p = 0.7, max_shift = 50)
    data = augmentation.gaussian_noise(data, p = 0.2, limit = 2.2)
    return data
```
# MyNet_0.8704
## Configs
```python
class cfg:
    seed = 'Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    model = 'EEGNet'
    activate = 'ReLU'
    optimizer = 'Ranger21'
    loss_function = 'cross_entropy'
    lr = 1e-2
    epoch = 300
```
```python
def train_transform(data):
    data = augmentation.time_shift(data, p = 0.7, max_shift = 50)
    data = augmentation.gaussian_noise(data, p = 0.2, limit = 2.2)
    return data
```
```python
class MyNet(nn.Module):
    def __init__(self, out_feature = 2, activate = cfg.activate):
        super().__init__()
        self.time_filter = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = (1, 51), stride = (1, 1), padding = (0, 25), bias = False),
            nn.BatchNorm2d(8, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        )
        self.depthWiseConv = nn.Sequential(
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
        self.separableConv = nn.Sequential(
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
        self.pooling = nn.AvgPool2d(kernel_size = (1, 5), stride = (1, 5), padding = 0)
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 960, out_features = out_feature, bias = True)
        )
    def activate_function(self, activate):
        activate_functions = {
            'ELU': nn.ELU(alpha = 1),
            'ReLU': nn.ReLU(),
            'Leaky ReLU': nn.LeakyReLU(),
            'Silu': nn.SiLU()
        }
        if activate in activate_functions:
            return activate_functions[activate]
        else:
            print('Activation function not found.')
    def forward(self, input):
        output = self.time_filter(input)
        output = self.shortcut_1(output) + self.depthWiseConv(output)
        output = self.pooling(output)
        output = self.shortcut_2(output) + self.separableConv(output)
        output = self.pooling(output)
        output = output.view(output.size(0), -1)    
        output = self.classifier(output)
        return output
```