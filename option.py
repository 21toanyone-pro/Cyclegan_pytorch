import argparse 
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataroot', type=str, default='datasets/data/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--img_size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--init_type", type=str, default='normal', help="normal | orth | xavier_uniform")
parser.add_argument("--gan_mode", type=str, default='lsgan', help="lsgan | projection | ")
parser.add_argument("--netD", type=str, default='basic', help="basic | projection | ")
parser.add_argument("--netG", type=str, default='Resnet', help="Resnet | projection | ")
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False