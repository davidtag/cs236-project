if __name__ == '__main__':
    import argparse
    import itertools

    import numpy as np
    import pdb
    
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    from PIL import Image
    import torch

    from models import CycleConsistentGenerator, Discriminator, iRevNetCycle
    from glow import Glow
    #from real_nvp import RealNVP
    from utils import ReplayBuffer
    from utils import LambdaLR
    from utils import Logger
    from utils import weights_init_normal
    from datasets import ImageDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

    # New args
    parser.add_argument('--generator', type=str, default='baseline', choices=['baseline', 'glow','realnvp','irevnet'])
    parser.add_argument('--identity_weight', type=float, default=5.0)
    parser.add_argument('--cycle_weight', type=float, default=10.0)
    parser.add_argument('--gan_weight', type=float, default=1)
    parser.add_argument('--scale_pixels', type=float, default=0.99)
    parser.add_argument('--g_clip_grad', type=float, default=0.)
    parser.add_argument('--d_clip_grad', type=float, default=0.)

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    if opt.generator == "baseline":
        generator = CycleConsistentGenerator(opt.input_nc, opt.output_nc)
        generator.apply(weights_init_normal)
    elif opt.generator == "glow":
        generator = Glow(16, opt.input_nc, 256, squeeze=4)
    elif opt.generator == "realnvp":
        generator = RealNVP()
    elif opt.generator == "irevnet":
        generator = iRevNetCycle([3,256,256])   

    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    if opt.cuda:
        generator.cuda()
        netD_A.cuda()
        netD_B.cuda()

    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=opt.g_lr, #betas=(0.5, 0.999),
                                   weight_decay=1e-3)
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.d_lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.d_lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                    transforms.RandomCrop(opt.size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader) // 10)
    ###################################

    def isfinite(x):
        """
        Quick pytorch test that there are no nan's or infs.

        note: torch now has torch.isnan
        url: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519
        """
        not_inf = ((x + 1) != x)
        not_nan = (x == x)
        return not_inf & not_nan


    ###### Training ######
    #pdb.set_trace()
    try:
        for epoch in range(opt.epoch, opt.n_epochs):
            for i, batch in enumerate(dataloader):
                
                ### MODEL INPUT #########################################
                A = (batch['A']+1)*0.5
                B = (batch['B']+1)*0.5
                
                real_A = Variable(input_A.copy_(A))
                real_B = Variable(input_B.copy_(B))

                
                ###### Generators A2B and B2A ######
                optimizer_G.zero_grad()

                if False:
                    real_A_copy = torch.tensor(real_A)
                    fake_B = generator.transform(real_A)
                    print("Real A diff: ", ((real_A-real_A_copy)**2).sum())

                    fake_B_ok = torch.tensor(fake_B)                
                    recovered_A_ok = generator.transform.inv(fake_B)
                    print("Fake B diff: ", ((fake_B-fake_B_ok)**2).sum()) 
                    loss_cycle_ABA_ok = criterion_cycle(recovered_A_ok, real_A)
                    print("Cycle loss: ", loss_cycle_ABA_ok) 

                # Identity loss
                # G_A2B(B) should equal B if real B is fed
                same_B = generator.transform(real_B)
                loss_identity_B = criterion_identity(same_B, real_B) * opt.identity_weight
                # G_B2A(A) should equal A if real A is fed
                same_A = generator.transform.inv(real_A)
                loss_identity_A = criterion_identity(same_A, real_A) * opt.identity_weight

                # GAN loss
                fake_B = generator.transform(real_A)
                pred_fake = netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

                fake_A = generator.transform.inv(real_B)
                pred_fake = netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

                # Cycle loss
                #print("Fake B diff: ", ((fake_B-fake_B_ok)**2).sum())
                recovered_A = generator.transform.inv(fake_B)
                #print("Rec A diff: ", ((recovered_A-recovered_A_ok)**2).sum())
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * opt.cycle_weight
                #print(loss_cycle_ABA)                

                recovered_B = generator.transform(fake_A)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B)  * opt.cycle_weight
                #cycle_loss = loss_cycle_ABA + loss_cycle_BAB
                #print(loss_cycle_BAB) 

                # Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()

                if opt.g_clip_grad > 0.:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), opt.g_clip_grad)

                optimizer_G.step()
                ###################################

                ###### Discriminator A ######
                optimizer_D_A.zero_grad()

                # Real loss
                pred_real = netD_A(real_A).view(target_real.shape)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)*0.5
                loss_D_A.backward()

                if opt.d_clip_grad > 0.:
                    torch.nn.utils.clip_grad_norm_(netD_A.parameters(), opt.d_clip_grad)
                optimizer_D_A.step()
                ###################################

                ###### Discriminator B ######
                optimizer_D_B.zero_grad()

                # Real loss
                pred_real = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)*0.5
                loss_D_B.backward()

                if opt.d_clip_grad > 0.:
                    torch.nn.utils.clip_grad_norm_(netD_B.parameters(), opt.d_clip_grad)
                optimizer_D_B.step()
                ###################################

                # Progress report (http://localhost:8097)
                print("\n------done batch")
                if True: #i % 10 == 0:
                    print(loss_G)
                    logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                                'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 
                                'loss_D': (loss_D_A + loss_D_B)},
                                images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()
            print("saving models")
            torch.save(generator.state_dict(), 'output/generator.pth')
            torch.save(netD_A.state_dict(), 'output/netD_A.pth')
            torch.save(netD_B.state_dict(), 'output/netD_B.pth')

    finally:
        # Save models checkpoints
        print("saving models")
        torch.save(generator.state_dict(), 'output/generator_final.pth')
        torch.save(netD_A.state_dict(), 'output/netD_A_final.pth')
        torch.save(netD_B.state_dict(), 'output/netD_B_final.pth')
    ###################################
