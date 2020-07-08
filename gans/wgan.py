import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

import time, datetime
import os

from libs import get_loader, initialize_weights, denorm


class G(nn.Module):
    
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(G, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        
        initialize_weights(self)
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)
        return x


class D(nn.Module):
    
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # no sigmoid
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
        )
        
        initialize_weights(self)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)
        return x


class WGAN:
    
    def __init__(self, args):
        
        # set parameters
        self.args = args
        self.z_dim = 62
        self.c = 0.01           # clipping grad value
        self.n_critic = 5       # iteraions of the critic per generator update
        
        # load dataloader
        self.data_loader = get_loader(args=args)
        data = self.data_loader.__iter__().__next__()[0]
        
        # net init
        self.G = G(input_dim=self.z_dim, output_dim=data.shape[1], input_size=args.input_size)
        self.D = D(input_dim=data.shape[1], output_dim=1, input_size=args.input_size)
        self.G = self.G.to(args.device)
        self.D = self.D.to(args.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        # fixed sample to visualize
        self.fixed_z = torch.rand((args.batch_size, self.z_dim))
        
        # tensorboard
        if args.use_tensorboard:
            from libs import Logger
            self.logger = Logger(log_dir=args.tb_path)
        
        # print model configuration
        print()
        
        print("##### Information #####")
        print("# gan type : ", args.gan_type)
        print("# dataset : ", args.dataset)
        print("# batch_size : ", args.batch_size)
        print("# epoch : ", args.epoch)
        
        print()
    
    def train(self):
        
        # Fixed z
        fixed_z = torch.rand((self.args.batch_size, self.z_dim))
        fixed_z = fixed_z.to(self.args.device)
        
        # Training...!
        self.G.train()
        self.D.train()
        total_it = 0
        start_time = time.time()
        loss_dict = {}
        print('Training start...!!')
        
        for epoch in range(self.args.epoch):
            
            for it, (x, _) in enumerate(self.data_loader):
                
                x = x.to(self.args.device)
                z = torch.rand((x.size(0), self.z_dim))
                z = z.to(self.args.device)

                # labels
                self.real_y = torch.ones(x.size(0), 1).to(self.args.device)
                self.fake_y = torch.zeros(x.size(0), 1).to(self.args.device)
                
                # ========== Update D ========== #
                D_real = self.D(x)
                real_D_loss = -torch.mean(D_real)
                
                fake_x = self.G(z)
                D_fake = self.D(fake_x)
                fake_D_loss = torch.mean(D_fake)
                
                D_loss = (real_D_loss + fake_D_loss) / 2
                
                self.D_optimizer.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()
                
                # clipping D
                for p in self.D.parameters():
                    p.data.clamp_(-self.c, self.c)
                
                loss_dict['D/real_D_loss'] = real_D_loss.item()
                loss_dict['D/fake_D_loss'] = fake_D_loss.item()
                
                # ========== Update G ========== #
                if (total_it + 1) % self.n_critic == 0:
                    fake_x = self.G(z)
                    D_fake = self.D(fake_x)
                    fake_G_loss = -torch.mean(D_fake)
                    
                    self.G_optimizer.zero_grad()
                    fake_G_loss.backward()
                    self.G_optimizer.step()
                    
                    loss_dict['G/fake_G_loss'] = fake_G_loss.item()

                if (total_it + 1) % self.args.log_freq == 0 or total_it == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = f"Elapsed [{et}], Epoch/Iter [{epoch + 1:03d}/{total_it + 1:07d}]"
                    for tag, value in loss_dict.items():
                        log += f", {tag}: {value:.4f}"
                    print(log)
    
                    if self.args.use_tensorboard:
                        for tag, value in loss_dict.items():
                            self.logger.scalar_summary(tag, value, total_it + 1)

                if (total_it + 1) % self.args.img_save_freq == 0 or total_it == 0:
                    self.G.eval()
    
                    with torch.no_grad():
                        fixed_fake_x = self.G(fixed_z)
                        img_path = os.path.join(self.args.img_path, f"{total_it + 1:07d}-images.png")
                        nrow = int(torch.sqrt(torch.Tensor([self.args.batch_size])).item())
                        save_image(denorm(fixed_fake_x.data.cpu()), img_path, nrow=nrow, padding=0)
    
                    self.G.train()

                if (total_it + 1) % self.args.ckpt_save_freq == 0 or total_it == 0:
                    G_path = os.path.join(self.args.ckpt_path, f'{total_it + 1:07d}-G.pth')
                    D_path = os.path.join(self.args.ckpt_path, f'{total_it + 1:07d}-D.pth')
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.D.state_dict(), D_path)

                total_it += 1




























































































































