import argparse
from pathlib import Path

# import sys
# print(sys.path)
# exit()

import torch

from gans import GAN, LSGAN, WGAN, WGAN_GP
from libs import str2bool, make_gif_with_samples



def parse_args():
    parser = argparse.ArgumentParser(description="Personal implementations of GAN")
    
    # Specify GAN type and dataset
    parser.add_argument('--desc', type=str, default=None, help='Experiment identifier')
    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['gan', 'lsgan', 'wgan', 'wgan_gp', 'dragan', 'ebgan', 'began'],
                        help='GAN TYPE')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn', 'stl10'],
                        help='DATASET')
    
    # Hyper-parameters
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=32, help='The size of input image')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    # Logging interval
    parser.add_argument('--use_tensorboard', type=str2bool, default=True, help='Whether use tensorboard')
    parser.add_argument('--ckpt_save_freq', type=int, default=5000, help='The number of iterations to save checkpoint')
    parser.add_argument('--img_save_freq', type=int, default=1000, help='The number of iterations to save images')
    parser.add_argument('--log_freq', type=int, default=10, help='The number of iterations to print logs')
    
    args = parser.parse_args()
    return process_args(args=args)

def create_folder_ifnotexist(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
    return path
    
def process_args(args):
    
    # if desc is none, make desc using gan_type and dataset
    if args.desc is None:
        args.desc = f'{args.gan_type}_{args.dataset}'
    
    # make directories
    print('Making directories...')
    STORAGE_PATH = create_folder_ifnotexist('./storage')
    IMG_PATH = create_folder_ifnotexist(STORAGE_PATH / 'images' / args.desc)
    CKPT_PATH = create_folder_ifnotexist(STORAGE_PATH / 'checkpoints'/ args.desc)
    
    args.img_path = IMG_PATH
    args.ckpt_path = CKPT_PATH
    
    if args.use_tensorboard:
        TB_PATH = create_folder_ifnotexist(STORAGE_PATH / 'tb' / args.desc)
        args.tb_path = TB_PATH

    # device
    is_gpu = torch.cuda.is_available()
    print(f'Using GPU:{is_gpu}')
    args.device = torch.device('cuda' if is_gpu else 'cpu')

    return args

def main():
    
    # parse arguments
    args = parse_args()
    
    # specify gan type
    if args.gan_type == 'gan':
        gan = GAN(args)
    elif args.gan_type == 'lsgan':
        gan = LSGAN(args)
    elif args.gan_type == 'wgan':
        gan = WGAN(args)
    elif args.gan_type == 'wgan_gp':
        gan = WGAN_GP(args)
    else:
        raise Exception(f" [!] There is no option for {args.gan_type}")
    
    # train gan
    gan.train()
    print(" [*] Training finished!")
    
    # visualize outputs
    make_gif_with_samples(args.img_path)
    print(" [*] Visualizing finished!")
    
if __name__ == '__main__':
    main()
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    





