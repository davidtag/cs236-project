if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch.distributions import transforms

    from models import Discriminator
    from utils import weights_init_normal,LambdaLR,ReplayBuffer

    import torchvision.transforms as tvf
    from torch.utils.data import DataLoader
    from datasets import ImageDataset
    
    from glow import Glow, TanhTransform

    ###########################################################################################
    def img2tensor(im_array,alpha=0.05,permute=True,batch_dim=True):
        """
        @param im_array: np.uint8 array-representation of an image of shape (h,w,c)
        @param alpha: scaling factor to avoid boundary effects
        @param permute: bring channel dimension to the fron of the image
        @param batch_dim: add a batch dimension to the front of the image

        @return: torch.float32 tensor of the image on an unbounded domain
                 achieved by passing bounded image through a logit (sigmoid^-1 function)
        """
        sigmoid = transforms.SigmoidTransform()                          #initialize sigmoid transformer
        noise   = np.random.uniform(size=im_array.shape)                 #random noise in [0,1)
        im_jittered = torch.tensor(im_array + noise,dtype=torch.float32) #jittered images in [0,256), convert to tensor
        im_norm = alpha+(1-alpha)*im_jittered/256                        #normalized image in (0,1)
        im_tensor = sigmoid.inv(im_norm)                                 #convert to unbounded numbers for modelling
        if permute:
            im_tensor = im_tensor.permute(2,0,1)                         #bring channel dim to front
        if batch_dim:
            im_tensor = im_tensor.unsqueeze(0)                           #add batch dim
        return im_tensor

    def tensor2img(im_tensor,alpha=0.05,was_permuted=True,has_batch_dim=True):
        """
        @param im_tensor: torch.float32 tensor of the image on an unbounded domain
        @param alpha: scaling factor used to generate tensor
        @param was_permuted: if True, channel dimension precedes spatial ones
        @param has_batch_dim: if True, first dimension is a batch one

        @return: the image as a np.uint8 array
                 that is first detached from the computational graph and converted
                 to a cpu tensor
        """
        sigmoid = transforms.SigmoidTransform()                 #initialize transformer
        if has_batch_dim:
            im_tensor = im_tensor.squeeze(0)                    #remove batch dim
        if was_permuted:
            im_tensor = im_tensor.permute(1,2,0)                #bring batch dim to back
        im_norm = sigmoid(im_tensor)
        im_jittered = 256*(im_norm - alpha)/(1-alpha)
        im_array = (im_jittered.detach().cpu().numpy()).astype(np.uint8)
        return im_array
    ###########################################################################################

    generator = Glow(16, 3, 32, squeeze=4)
    netD_A = Discriminator(3)
    netD_B = Discriminator(3)

    generator.cuda()
    netD_A.cuda()
    netD_B.cuda()

    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)


    criterion_GAN = torch.nn.MSELoss()
    criterion_identity = torch.nn.MSELoss() #torch.nn.L1Loss()


    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=0.00001, betas=(0.5, 0.999),
                                   weight_decay=1e-2)
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(200, 0, 100).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(200, 0, 100).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(200, 0, 100).step)


    Tensor = torch.cuda.FloatTensor if True else torch.Tensor
    input_A = Tensor(1, 3, 128, 128)
    input_B = Tensor(1, 3, 128, 128)
    target_real = Variable(Tensor(1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()


    transforms_ = [tvf.Resize(int(128*1.12), Image.BICUBIC),
                   tvf.RandomCrop(128),
                   tvf.RandomHorizontalFlip(),
                  ]
    dataloader = DataLoader(ImageDataset('datasets/facades/', transforms_=transforms_, unaligned=True),
                            batch_size=1, shuffle=True, num_workers=4)


    t = TanhTransform()
    u = torch.distributions.Uniform(-0.3,0.3)


    save_counter = 0
    for epoch in range(100):
        print("--Starting Epoch {}".format(epoch))
        for i, batch in enumerate(dataloader):
            ###### Input ######################
            A = batch['A']
            B = batch['B']

            real_A = Variable(input_A.copy_(A))
            real_B = Variable(input_B.copy_(B))


            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            same_B = generator.transform(real_B)       # G_A2B(B) should equal B if real B is fed
            same_A = generator.transform.inv(real_A)   # G_B2A(A) should equal A if real A is fed
            loss_identity = (criterion_identity(same_B, real_B) + criterion_identity(same_A, real_A))*0.5

            #same_A = generator.transform(real_A)       # G_A2B(B) should equal B if real B is fed
            #same_B = generator.transform.inv(real_B)   # G_B2A(A) should equal A if real A is fed
            #loss_identity_2 = criterion_identity(same_B, real_B) + criterion_identity(same_A, real_A)    

            # GAN loss
            fake_B = generator.transform(real_A)
            pred_fake = netD_B(t(fake_B))
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real+u.rsample().to('cuda'))

            fake_A = generator.transform.inv(real_B)
            pred_fake = netD_A(t(fake_A))
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real+u.rsample().to('cuda'))

            loss_GAN = loss_GAN_A2B + loss_GAN_B2A

            #Total Loss
            loss_G = loss_GAN + 12*loss_identity #+ 0.5*loss_identity_2
            loss_G.backward()

            torch.nn.utils.clip_grad_value_(generator.parameters(),2)

            optimizer_G.step()

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(t(real_A))  #.view(target_real.shape)
            loss_D_real = criterion_GAN(pred_real, target_real+u.rsample().to('cuda'))

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(t(fake_A.detach()))
            loss_D_fake = criterion_GAN(pred_fake, target_fake+u.rsample().to('cuda'))

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()


            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(t(real_B))
            loss_D_real = criterion_GAN(pred_real, target_real+u.rsample().to('cuda'))

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(t(fake_B.detach()))
            loss_D_fake = criterion_GAN(pred_fake, target_fake+u.rsample().to('cuda'))

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()


            ###### Log ######
            print("Iteration {}: loss_G_identity = ({:.5f},{:.5f}), loss_G_GAN = {:.5f}, loss_D_A = {:.5f}, loss_D_B = {:.5f}".format(i,loss_identity,0,loss_GAN,loss_D_A,loss_D_B))

            if i % 20 == 0: 
                torch.save(generator.state_dict(), 'output/glow/{}-{}-generator.pth'.format('facades',save_counter))
                torch.save(netD_A.state_dict(), 'output/glow/{}-{}-netD_A.pth'.format('facades',save_counter))
                torch.save(netD_B.state_dict(), 'output/glow/{}-{}-netD_B.pth'.format('facades',save_counter))
                save_counter += 1
