# EEG ELU
## EEG_ELU_0.839
path : /home/pp037/DeepLearning_NYCU/Lab_2/model/EEG_ELU_0.839.pkl
```python
class cfg:
    seed = 'Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    activate = 'ELU'
    lr = 1e-3
    min_lr = 1e-7
    T0 = 20
    epoch = 100
```
## EEG_ELU_0.857
- path : /home/pp037/DeepLearning_NYCU/Lab_2/model/EEG_ELU_0.857.pkl
```python
class cfg:
    seed = 'https://github.com/Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    activate = 'ELU'
    lr = 1e-3
    min_lr = 1e-7
    T0 = 30
    epoch = 180
```
# EEG ReLU
## EEG_ReLU_0.893
- path : /home/pp037/DeepLearning_NYCU/Lab_2/model/EEG_ReLU_0.893.pkl
```python
class cfg:
    seed = 'Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    activate = 'ReLU'
    lr = 1e-3
    min_lr = 1e-7
    T0 = 40
    epoch = 200
```