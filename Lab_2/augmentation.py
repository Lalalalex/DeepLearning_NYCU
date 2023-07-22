import numpy as np

def channel_swap(data, p = 0.7):
    swapped_data = data
    if np.random.rand() >= p:
        swapped_data[:, 0, :] = data[:, 1, :]
        swapped_data[:, 1, :] = data[:, 0, :]
    
    return swapped_data

def time_shift(data, p = 0.5, max_shift = 50):
    num_channels = data.shape[1]
    shifted_data = data
    if np.random.rand() <= p:
        for channel in range(num_channels):
            shift_amount = np.random.randint(-max_shift, max_shift + 1)
            shifted_data[:, channel, :] = np.roll(data[:, channel, :], shift_amount, axis = -1)
    
    return shifted_data

def gaussian_noise(data, p = 0.2, limit = 1):
    sign = np.random.choice([-1, 1], size = data.shape)
    noise = np.random.normal(0, 1, size = data.shape) * limit * sign
    if np.random.rand() <= p:
        return data + noise
    else:
        return data
    noise = np.random.normal(0, 1, size = data.shape) * limit
    if np.random.rand() <= p:
        return data + noise
    else:
        return data

def tima_reverse(data, p = 0.5):
    flip_data = np.flip(data, axis = -1).copy()
    if np.random.rand() <= p:
        return flip_data
    else:
        return data

def random_eliminate(data, p = 0.2, max_eliminate = 15):
    if np.random.rand() <= p:
        num_eliminates = np.random.randint(low = 0, high = max_eliminate + 1)
        eliminate_index = np.random.randint(low = 0, high = data.shape[2], size = num_eliminates)
        for i in eliminate_index:
            data[:, int(np.round(np.random.rand())),i] = 0
    return data