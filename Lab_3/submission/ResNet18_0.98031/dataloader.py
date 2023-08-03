import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import cv2
from cfg import cfg

base_path = 'data'
base_data_path = os.path.join(base_path, 'new_dataset')

def get_image_path(path, base_data_path = base_data_path):
    image_path = os.path.join(base_data_path, path.split('./')[-1])
    return image_path
def get_image(image_path):
    image = cv2.imread(image_path)
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_RGB
def showExample(df, random = True, sample_num = 5):
    classes = set(df['label'].values)
    class_count = len(classes)
    fig, ax = plt.subplots(class_count, 5, figsize = (5 * sample_num, 5 * class_count))
    for index, label in enumerate(classes):
        if random:
            sample = df[df['label'] == label].sample(sample_num).reset_index(drop = True)
        else:
            sample = df[df['label'] == label].sample(sample_num, random_state = seed).reset_index(drop = True)
        for i in sample.index:
            image_id = sample.iloc[i]['Path']
            image = get_image(get_image_path(image_id))
            image_id = image_id.split('/')[-1].split('.')[0]
            ax[index,i].set_title(str(image_id + '_label_' + str(label)))
            ax[index,i].axis('off')
            ax[index,i].imshow(image)
            plt.savefig('image_example')

class LeukemiaDataset(Dataset):
    def __init__(self, df, is_test_model = False, transforms = None):
        self.df = df
        self.is_test_model = is_test_model
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        image_id = self.df.iloc[index]['Path']
        image_path = get_image_path(image_id)
        image = get_image(image_path)
        if self.transforms:
            image = self.transforms(image = image)['image']
        if self.is_test_model:
            return image_id, image
        label = self.df.iloc[index]['label']
        return image_id, image, label

def split_dataset(df, split = 0.8):
    df = df.sample(frac = 1).reset_index(drop = True)
    train_df = df.iloc[:int(len(df)*split)]
    valid_df = df.iloc[int(len(df)*split):]
    return train_df, valid_df

train_df_path = os.path.join(base_path, 'train.csv')
valid_df_path = os.path.join(base_path, 'valid.csv')
test_df_path = os.path.join(base_path, 'resnet_18_test.csv')
all_data_df = pd.concat([pd.read_csv(train_df_path), pd.read_csv(valid_df_path)])
test_df = pd.read_csv(test_df_path)

train_df, valid_df = split_dataset(all_data_df, cfg.split)