import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

import cfg
from cfg import cfg
from cfg import send_message
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def show_curve(data, title = 'PSNR', x_label = 'frame', y_label = 'psnr', file_name = 'val_psnr'):
    plt.figure()
    plt.title(title)
    for i in data:
        plt.plot(data[i], label = i)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(file_name)

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    KLD /= batch_size
    if torch.isnan(KLD) or torch.isinf(KLD):
        send_message('nan')
    return KLD

class kl_annealing():
    def __init__(self, args, current_epoch = 0, kl_type = "N"):
        self.args = args
        self.current_epoch = current_epoch
        self.step = 0
        self.n_per_cycle, self.beta_rate = self.frange_cycle_linear\
        (cycle = self.args.kl_anneal_cycle, ratio = self.args.kl_anneal_ratio)
        if kl_type == "N":
            self.kl_type = args.kl_anneal_type
        else:
            self.kl_type = kl_type
        
    def update(self):
        self.current_epoch += 1
        self.step = self.current_epoch % self.n_per_cycle
    
    def get_beta(self):
        if self.kl_type == "None":
            return 1.0
        elif self.kl_type == "Cyclical":
            return min(1.0, self.step * self.beta_rate)
        elif self.kl_type == "Monotonic":
            return self.monotonic()

    def monotonic(self, max_beta = 1.0, beta_rate = 0.1):
        return min(max_beta, beta_rate * self.current_epoch)

    def frange_cycle_linear(self, min_beta = 0.0, max_beta = 1.0, cycle = 10, ratio = 1.0):
        beta_range = max_beta - min_beta
        n_per_cycle = self.args.num_epoch/cycle
        beta_rate = beta_range/((n_per_cycle - 1) * ratio)
        return n_per_cycle, beta_rate

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr)
        #self.scheduler = CosineAnnealingWarmupRestarts(self.optim, first_cycle_steps = 7, cycle_mult = 1.0, max_lr=0.01, min_lr = 0.0001, warmup_steps=3, gamma=0.8)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch = 0)

        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        self.best_psnr = 0
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            if self.current_epoch - 1 % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            self.save(os.path.join(self.args.save_root, f"latest.ckpt"))
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing, self.kl_annealing)
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TF: ON, {:.1f}], beta: {:.3f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TF: OFF, {:.1f}], beta: {:.3f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
       
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_cycle.update()

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label).detach().cpu()
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

    
    def training_one_step(self, img, label, adapt_TeacherForcing, kl_annealing):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)

        kld = 0
        mse = 0
        out = img[0]

        for i in range(1, self.train_vi_len):
            label_feat = self.label_transformation(label[i])
            human_feat_hat = self.frame_transformation(img[i])
            if adapt_TeacherForcing:
                last_human_feat = self.frame_transformation(img[i - 1])\
                * 0.3 + self.frame_transformation(out) * 0.7
            else:
                last_human_feat = self.frame_transformation(out)

            z, mu, logvar = self.Gaussian_Predictor(human_feat_hat, label_feat)
            parm = self.Decoder_Fusion(last_human_feat, label_feat, z)
            out = self.Generator(parm)

            kld = kld + kl_criterion(mu, logvar, self.batch_size)
            mse = mse + self.mse_criterion(img[i], out)
        
        beta = kl_annealing.get_beta()
        loss = mse + beta*kld
        
        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()

        return loss
    
    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)

        decoded_frame_list = [img[0].cpu()]
        label_list = []

        kld = 0
        mse = 0
        total_psnr = 0
        avg_psnr = 0

        last_human_feat = self.frame_transformation(img[0])
        first_templete = last_human_feat.clone()
        out = img[0]

        for i in range(1, self.val_vi_len):
            label_feat = self.label_transformation(label[i])
            human_feat_hat = self.frame_transformation(out)
            #z, mu, logvar = self.Gaussian_Predictor(human_feat_hat, label_feat)
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()

            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)
            out = self.Generator(parm)
            
            decoded_frame_list.append(out.cpu())
            label_list.append(label[i].cpu())
            
            psnr = Generate_PSNR(img[i], out).cpu()
            total_psnr = total_psnr + psnr
            
            mse = mse + self.mse_criterion(out, img[i])
        
        loss = mse
        avg_psnr = total_psnr/self.val_vi_len

        if avg_psnr > self.best_psnr:
            send_message('best_update')
            self.best_psnr = avg_psnr
            self.save(os.path.join(self.args.save_root, 'best.ckpt'))

        print('PSNR: ', avg_psnr.item())
        send_message(str(avg_psnr))
            
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        label_frame = stack(label_list).permute(1, 0, 2, 3, 4)
        
        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'val.gif'))

        return loss
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(0.0, self.tfr - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.5f}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer" : self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       : self.tfr,
            "last_epoch": self.current_epoch,
            "best_psnr" : self.best_psnr
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")

    args = parser.parse_args()
    
    main(args)