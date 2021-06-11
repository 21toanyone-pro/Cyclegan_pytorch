from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np 
from PIL import Image
import itertools
from tqdm import tqdm
import option
from dataset import ImageDataset
from network import Generators, Discriminators, define_D, define_G
from torch.autograd import Variable
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init
from torchvision.utils import save_image
from cycle_gan_model import CycleGAN


if __name__ =='__main__':
    opt = option.opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data load
    transforms_ = [ transforms.Resize(int(opt.img_size), Image.BICUBIC), 
                transforms.RandomCrop(opt.img_size), 
                #transforms.RandomHorizontalFlip(),       
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batch_size, shuffle=True, num_workers=0)

    # 1) Network
    netG = define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, netG = opt.netG)#Generators(in_channels=opt.input_nc, out_channels=opt.output_nc)
    netF = define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, netG = opt.netG)#Generators(in_channels=opt.input_nc, out_channels=opt.output_nc)
    netDy = define_D(input_nc=opt.input_nc, output_nc=opt.output_nc, netD = opt.netD)#Discriminators(in_channels=opt.input_nc)
    netDx = define_D(input_nc=opt.input_nc, output_nc=opt.output_nc, netD = opt.netD)#Discriminators(in_channels=opt.input_nc)
    netG.cuda()
    netF.cuda()
    netDy.cuda()
    netDx.cuda()

    # 2) weights initialize 
    netG.apply(weights_init)
    netF.apply(weights_init)
    netDy.apply(weights_init)
    netDx.apply(weights_init)

    # 3) define loss
    ganLoss = CycleGAN.GANloss(gan_mode=opt.gan_mode)
    cycleConsistencyLoss = nn.L1Loss()
    identityLoss = nn.L1Loss()

    # 4) Optimizers & LR schedulers
    gen_optimizer = torch.optim.Adam(itertools.chain(netG.parameters(), netF.parameters()),
                                        opt.lr, (opt.beta1, 0.999))
    disDx_optimizer = torch.optim.Adam(netDx.parameters(),opt.lr, (opt.beta1, 0.999))
    disDy_optimizer = torch.optim.Adam(netDy.parameters(),opt.lr, (opt.beta1, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(disDx_optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(disDy_optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor
    target_real = Tensor(opt.batch_size).fill_(1.0) # label for a real image
    target_fake = Tensor(opt.batch_size).fill_(0.0) # label for a fake image

    fake_A_buffer = ReplayBuffer() 
    fake_B_buffer = ReplayBuffer()

    for epoch in range(0, opt.n_epochs):
        for i, img in enumerate(dataloader):
            # model input
            real_A, real_B, A_label, B_label = CycleGAN.set_input(img, device)
            ###############       GEN      ##################
            gen_optimizer.zero_grad()
            
            ## s
            #identity loss
            #netG(B) should equal B if real B is fed
            same_B = netG(real_B)
            loss_identity_B = identityLoss(same_B, real_B)*5.0
            #netF(A) should equal B if real A is fed
            same_A = netF(real_A)
            loss_identity_A = identityLoss(same_A, real_A)*5.0

            #gan loss
            fake_B = netG(real_A)
            save_image(fake_B, f'./checkpoint/gen_data/{epoch}_fake_B_.jpg', nrow=5, normalize=True, scale_each=True)
            
            pred_fakeDy = netDy(fake_B)
            loss_GAN_A2B = ganLoss(pred_fakeDy, target_real)

            fake_A = netF(real_B)
            save_image(fake_A, f'./checkpoint/gen_data/{epoch}_fake_A_.jpg', nrow=5, normalize=True, scale_each=True)
            pred_fakeDx = netDx(fake_A)
            loss_GAN_B2A = ganLoss(pred_fakeDx, target_real)

            #Cycleloss 
            recon_A = netF(fake_B)
            loss_Cycle_ABA = cycleConsistencyLoss(recon_A, real_A)*10.0

            recon_B = netG(fake_A)
            loss_Cycle_BAB = cycleConsistencyLoss(recon_B, real_B)*10.0

            #Total lossG
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B+loss_GAN_B2A+loss_Cycle_ABA+loss_Cycle_BAB
            loss_G.backward()
            gen_optimizer.step()
            ###############       GEN      ##################

            ###############       Dis_A      ##################
            disDx_optimizer.zero_grad()

            #real loss
            pred_real = netDx(real_A)
            loss_D_real = ganLoss(pred_real, target_real)

            #Fake loss 
            #create image buffer to store previously generated images
            rfake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netDx(rfake_A.detach())
            loss_D_fake = ganLoss(pred_fake, target_fake)

            #total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()
            disDx_optimizer.step()
            ###############       Dis_A      ##################

            ###############       Dis_B      ##################
            disDy_optimizer.zero_grad()

            #real loss
            pred_real = netDy(real_B)
            loss_D_real = ganLoss(pred_real, target_real)
            
            #Fake loss
            rfake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netDy(rfake_B.detach())
            loss_D_fake = ganLoss(pred_fake, target_fake)
            #total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()
            disDy_optimizer.step()


        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f][loss_G_identity: %f]['loss_G_GAN: %f]['loss_G_cycle: %f]" %
                (epoch, opt.n_epochs, i % len(dataloader), len(dataloader), (loss_D_A + loss_D_B), loss_G, (loss_identity_A + loss_identity_B), 
                (loss_GAN_A2B + loss_GAN_B2A), (loss_Cycle_ABA + loss_Cycle_BAB)))
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG.state_dict(), 'output/netG_A2B.pth')
        torch.save(netF.state_dict(), 'output/netG_B2A.pth')
        torch.save(netDx.state_dict(), 'output/netD_A.pth')
        torch.save(netDy.state_dict(), 'output/netD_B.pth')
            




