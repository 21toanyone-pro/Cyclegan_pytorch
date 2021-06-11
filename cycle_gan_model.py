import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class CycleGAN():
    def set_input(input, device):
        real_A = input['A' ].to(device)
        real_B = input['B'].to(device)
        real_A_label = input['A_label'].to(device)
        real_B_label = input['B_label'].to(device)
        return real_A, real_B, real_A_label, real_B_label

    def GANloss(gan_mode):
        if gan_mode == 'lsgan':
            loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            loss = nn.BCEWithLogitsLoss()

        #수정중
        elif gan_mode == 'projection':
            loss = nn.MSELoss()
        return loss

