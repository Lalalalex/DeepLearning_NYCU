# EEGNet_ReLU_
- path:/home/pp037/DeepLearning_NYCU/Lab_2/model/EEGNet_ReLU_0.8722.pkl
## Configs
```python
class cfg:
    seed = 'Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    activate = 'ReLU'
    lr = 5e-2
    epoch = 300
```
```python
def train_transform(data):
    data = augmentation.time_shift(data, p = 0.7, max_shift = 50)
    data = augmentation.gaussian_noise(data, p = 0.2, limit = 2.2)
    return data
```
