# EEG ELU
## EEG_ELU_0.9286
path: /home/pp037/DeepLearning_NYCU/Lab_2/model/EEG_ELU_0.9286.pkl
```python
class cfg:
    seed = 'Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    activate = 'ELU'
    lr = 5e-3
    min_lr = 5e-8
    T0 = 30
    epoch = 200
def time_shift(data, max_shift = 50, p = 1):
    num_channels = data.shape[1]
    shifted_data = data
    if np.random.rand() <= p:
        for channel in range(num_channels):
            shift_amount = np.random.randint(-max_shift, max_shift + 1)
            shifted_data[:, channel, :] = np.roll(data[:, channel, :], shift_amount, axis = -1)
    
    return shifted_data
```
# EEG ReLU
## EEG_ReLU_0.9643

```python
class cfg:
    seed = 'https://github.com/Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    activate = 'ReLU'
    lr = 5e-3
    min_lr = 5e-8
    T0 = 50
    epoch = 200
def time_shift(data, max_shift = 50, p = 1):
    num_channels = data.shape[1]
    shifted_data = data
    if np.random.rand() <= p:
        for channel in range(num_channels):
            shift_amount = np.random.randint(-max_shift, max_shift + 1)
            shifted_data[:, channel, :] = np.roll(data[:, channel, :], shift_amount, axis = -1)
    
    return shifted_data

def gaussian_noise(data, p = 0.5, limit = 2):
    noise = np.random.normal(0, 1, size = data.shape) * limit
    if np.random.rand() <= p:
        return data + noise
    else:
        return data
```
## EEG_ReLU_0.9464
- path: /home/pp037/DeepLearning_NYCU/Lab_2/model/EEG_ReLU_0.9464.pkl
```python
class cfg:
    seed = 'https://github.com/Lalalalex'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    activate = 'ReLU'
    lr = 5e-3
    min_lr = 5e-8
    T0 = 50
    epoch = 200
def time_shift(data, max_shift = 50, p = 1):
    num_channels = data.shape[1]
    shifted_data = data
    if np.random.rand() <= p:
        for channel in range(num_channels):
            shift_amount = np.random.randint(-max_shift, max_shift + 1)
            shifted_data[:, channel, :] = np.roll(data[:, channel, :], shift_amount, axis = -1)
    
    return shifted_data
```

# EEG Leaky ReLU
## EEG_Leaky_ReLU_0.949
- path: /home/pp037/DeepLearning_NYCU/Lab_2/model/EEG_Leaky_ReLU_0.949.pkl
```python
class cfg:
    seed = 'https://github.com/Lalalalexmi'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = 0.9
    batch_size = 128
    activate = 'Leaky ReLU'
    lr = 5e-3
    min_lr = 5e-8
    T0 = 30
    epoch = 200

def time_shift(data, max_shift = 50):
    _, num_channels, signal_length = data.shape
    shifted_data = np.zeros_like(data)
    
    for ch in range(num_channels):
        shift_amount = np.random.randint(-max_shift, max_shift + 1)
        shifted_data[:, ch, :] = np.roll(data[:, ch, :], shift_amount, axis = -1)
    
    return shifted_data
```