import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.transforms import Transform

from iRevNet import *

######################################################################################
class iRevNetCycle(nn.Module):
    def __init__(self,in_shape=[3, 224, 224]):
        super(iRevNetCycle,self).__init__()
        self.model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2,2,2,2],
                             nChannels=[24,96,384,1536], nClasses=1000, init_ds=2,
                             dropout_rate=0., affineBN=True, in_shape=in_shape,
                             mult=4)
        self.transform = CycleConsistentTransform(self.model.forward,
                                                 self.model.inverse)

######################################################################################        
class CycleConsistentGenerator(nn.Module):
    def __init__(self, a_nc, b_nc, g_n_residual_blocks=9):
        super(CycleConsistentGenerator, self).__init__()
        self.netG_A2B = Generator(a_nc, b_nc, g_n_residual_blocks)
        self.g_backward = Generator(b_nc, a_nc, g_n_residual_blocks)

        self.transform = CycleConsistentTransform(self.netG_A2B,
                                                  self.g_backward)
        
        
######################################################################################        
class CycleConsistentTransform(Transform):
    bijective = False
    def __init__(self, f, i):
        super(CycleConsistentTransform, self).__init__()
        self.f = f
        self.i = i

    def _call(self, x):
        return self.f(x)

    def _inverse(self, x):
        return self.i(x) 
    
######################################################################################        
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        z = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return z.squeeze(dim=1)
