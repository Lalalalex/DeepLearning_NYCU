import torch
import torch.utils.data
import torchvision
from PIL import Image
import argparse
import warnings
warnings.filterwarnings('ignore')
from diffusers import DDPMScheduler, UNet2DModel
from model import MyConditionedUNet
from dataloader import ICDataset
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import json
import os
from evaluator import evaluation_model
import torchvision.transforms as transforms
from cfg import send_message
import torch.nn as nn
from cfg import setSeed

class Trainer():
    def __init__(self, dataset, eps_model, noise_scheduler, device, args):
        self.dataset = dataset
        self.device = device

        self.epochs = args.epoch
        self.n_samples = args.n_samples
        self.image_channel = args.image_channel
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.save_image_path = args.save_image_path
        
        self.best_accuracy = -10
    
        self.eps_model = eps_model.to(self.device)
        self.noise_scheduler = noise_scheduler
        self.loss_fuction = nn.MSELoss()

        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True)
        self.optimizer = torch.optim.AdamW(self.eps_model.parameters(), lr = self.learning_rate)
        
    def get_test_label(self, data_path = "test.json"):
        label_dict = json.load(open(os.path.join('/home/pp037/DeepLearning_NYCU/Lab_6/data', "objects.json")))
        labels = json.load(open(os.path.join('/home/pp037/DeepLearning_NYCU/Lab_6/data', data_path)))
        
        newLabels = []
        for i in range(len(labels)):
            onehot_label = torch.zeros(24, dtype=torch.float32)
            for j in range(len(labels[i])):
                onehot_label[label_dict[labels[i][j]]] = 1 
            newLabels.append(onehot_label)
        
        return newLabels

    def save_images(self, images, name):
        grid = torchvision.utils.make_grid(images)
        save_image(grid, fp = self.save_image_path + name +".png")
    
    def transform(self):
        return transforms.Compose([
            transforms.Normalize((0, 0, 0), (1/0.5, 1/0.5, 1/0.5)),
            transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
        ])

    def sample(self, epoch = 0):
        test_label = torch.stack(self.get_test_label()).to(self.device)
        new_test_label = torch.stack(self.get_test_label(data_path = 'new_test.json')).to(self.device)
        
        x_test = torch.randn([len(test_label), self.image_channel, self.image_size, self.image_size], device=self.device)
        x_new_test = torch.randn([len(new_test_label), self.image_channel, self.image_size, self.image_size], device=self.device)
        with tqdm(self.noise_scheduler.timesteps, unit = 'Step', desc = 'Test') as tqdm_loader:
            for index, t in enumerate(tqdm_loader):
                with torch.no_grad():
                    residual_test = self.eps_model(x_test, t, test_label).sample
                    residual_new_test = self.eps_model(x_new_test, t, new_test_label).sample
                x_test = self.noise_scheduler.step(residual_test, t, x_test).prev_sample  
                x_new_test = self.noise_scheduler.step(residual_new_test, t, x_new_test).prev_sample         
        
        test_label = torch.tensor(test_label, dtype = torch.float32)
        new_test_label = torch.tensor(new_test_label, dtype = torch.float32)
        image = (x_test / 2 + 0.5).clamp(0, 1)
        new_image = (x_new_test / 2 + 0.5).clamp(0, 1)
        evaluate = evaluation_model()
        test_accuracy = evaluate.eval(x_test, test_label)
        new_test_accuracy = evaluate.eval(x_new_test, new_test_label)
        print('Test: ' + str(test_accuracy))
        print('New Test: '+ str(new_test_accuracy))
        accuracy = 0.5 * new_test_accuracy + 0.5 * test_accuracy

        if accuracy >= self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.eps_model,'Best_UNet_model.pkl')
            send_message('Best model update. Averager accuracy = ' + str(accuracy) + '.')
        save_image(make_grid(image, nrow=8), "{}/{}_{}.png".format('./image', 'test', str(epoch)))
        save_image(make_grid(new_image, nrow=8), "{}/{}_{}.png".format('./image', 'new_test', str(epoch)))

    def train_epoch(self):
        total_loss = 0
        with tqdm(self.data_loader, unit = 'Batch', desc = 'Train') as tqdm_loader:
            for index, (image, label) in enumerate(tqdm_loader):
                image = image.to(self.device)
                label = torch.tensor(label.to(device = self.device), dtype = torch.float32)

                noise = torch.randn_like(image).to(self.device)
                timesteps = torch.randint(0, 999, (image.shape[0],)).long().to(self.device)
                noise_image = self.noise_scheduler.add_noise(image, noise, timesteps)
                predict_noise = self.eps_model(noise_image, timesteps, label).sample

                loss = self.loss_fuction(predict_noise, noise)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.eps_model.parameters(), 5.)
                self.optimizer.step()

                total_loss = total_loss + loss.detach().cpu()
                average_loss = total_loss/(index + 1)
                tqdm_loader.set_postfix(Average_loss = average_loss.item())

    def train(self):
        for epoch in range(self.epochs):
            print('Epoch: ' + str(epoch + 1))
            self.train_epoch()
            if epoch % 5 == 0:
                self.sample(epoch)

    def test(self, model_path = None):
        if model_path != None:
            torch.load(self.eps_model, model_path)
        self.sample()

def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--test_only', action = 'store_true')
    parser.add_argument('--model_path', default = './Best_UNet_model.pkl')
    parser.add_argument('--data_path', default = './data')
    parser.add_argument('--save_image_path', default = './image')
    parser.add_argument('--seed', default = 'Lalalalex')
    parser.add_argument('--epoch', default = 100, type = int)
    parser.add_argument('--n_samples', default = 64, type = int)
    parser.add_argument('--train_time_steps', default = 1000, type = int)
    parser.add_argument('--image_channel', default = 3, type = int)
    parser.add_argument('--image_size', default = 64, type = int)
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--learning_rate', default = 1e-4, type = float)
    args = parser.parse_args()

    setSeed(args.seed)
    
    eps_model = MyConditionedUNet(
        sample_size=64,  
        in_channels = 3,                
        out_channels = 3,
        layers_per_block = 2,
        block_out_channels = (64, 128, 256, 256, 512, 512),
        down_block_types = (
            "DownBlock2D",         
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",      
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",      
            "UpBlock2D",           
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        class_embed_type='timestep',)
    train_dataset = ICDataset(root = args.data_path)
    noise_scheduler = DDPMScheduler(num_train_timesteps = args.train_time_steps, beta_schedule='squaredcos_cap_v2')
    trainer = Trainer(train_dataset, eps_model, noise_scheduler, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), args)
    
    if not args.test_only:
        trainer.train()
    trainer.test()

if __name__ == '__main__':
    main()