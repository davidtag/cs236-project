"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR, 2018
(c) Joern-Henrik Jacobsen, 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model_utils import split, merge, injective_pad, psi
from torch.distributions import transforms



class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=4):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.pad = 2 * out_ch - in_ch
        self.stride = stride
        self.inj_pad = injective_pad(self.pad)
        self.psi = psi(stride)
        if self.pad != 0 and stride == 1:
            in_ch = out_ch * 2
            print('')
            print('| Injective iRevNet |')
            print('')
        layers = []
        ###########################
        if not first:
            layers.append(nn.BatchNorm2d(in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_ch//2, int(out_ch//mult), kernel_size=3,
                      stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch//mult), int(out_ch//mult),
                      kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch//mult), out_ch, kernel_size=3,
                      padding=1, bias=False))
        ###########################
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        #pdb.set_trace()
        if self.pad != 0 and self.stride == 1:
            x = merge(x[0], x[1])
            x = self.inj_pad.forward(x)
            x1, x2 = split(x)
            x = (x1, x2)
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        if self.pad != 0 and self.stride == 1:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class iRevNet(nn.Module):
    def __init__(self, nBlocks, nStrides, nClasses, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4):
        super(iRevNet, self).__init__()
        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        print('')
        print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3 + 1))
        if not nChannels:
            nChannels = [self.in_ch//2, self.in_ch//2 * 4,
                         self.in_ch//2 * 4**2, self.in_ch//2 * 4**3]

        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)
        #self.bn1 = nn.BatchNorm2d(nChannels[-1]*2, momentum=0.9)
        #self.linear = nn.Linear(nChannels[-1]*2, nClasses)
        
        self.t = transforms.SigmoidTransform()

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, x):
        """ irevnet forward """
        (batch_in,channels_in,width_in,height_in) = x.shape
        x = self.t.inv(x*0.9999999+0.000000001)
        #########################################################
        n = self.in_ch//2
        if self.init_ds != 0:
            x = self.init_psi.forward(x)
        out = (x[:, :n, :, :], x[:, n:, :, :])
        for block in self.stack:
            out = block.forward(out)
        out_bij = merge(out[0], out[1])
        
        ##stop here
        #out = F.relu(self.bn1(out_bij))
        #out = F.avg_pool2d(out, self.ds)
        #out = out.view(out.size(0), -1)
        #out = self.linear(out)
        #return out, out_bij
        #########################################################
        #return out_bij,out_bij.view(batch_in,channels_in,width_in,height_in)
        #print(out_bij.shape)
        
        #current_out = self.reshape(out_bij,dims=(batch_in,channels_in,width_in,height_in),direction='forward')
        #back_opt1 = current_out.view(1,3072,8,8)
        
        
        #print("loss = {}".format(  ((back_opt1-out_bij)**2).sum()  ))
        
        #pdb.set_trace()
        #########################################################
        out = self.t(out_bij)
        out_reshape = self.reshape(out,dims=(batch_in,channels_in,width_in,height_in),direction='forward')
        
        return out_reshape
        
    def inverse(self, out_bij):
        """ irevnet inverse """
        out_bij = self.reshape(out_bij,dims=(1,3072,8,8),direction='backward')
        out_bij = self.t.inv(out_bij*0.9999999+0.000000001)
        ######################################
        out = split(out_bij)
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)
        out = merge(out[0],out[1])
        if self.init_ds != 0:
            x = self.init_psi.inverse(out)
        else:
            x = out
        #return x
        ######################################
        x = self.t(x)
        return x
    
    def reshape(self,x,dims,direction='forward'):
        if direction == 'forward':
            return x.contiguous().view(*dims)
        elif direction == 'backward':
            return x.contiguous().view(*dims)
        else:
            raise ValueError('direction should be forward or backward')


            

if __name__ == '__main__':
    ########################################################################
    #CODE TO TEST IMPLEMENTATION
    
    #import pdb
    #pdb.set_trace()
    device = torch.device('cuda')
    
    ########################################################################
    #Initialize Model
    model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2,2,2,2],
                    nChannels=[24,96,384,1536], nClasses=1000, init_ds=2,
                    dropout_rate=0., affineBN=True, in_shape=[3, 256, 256],
                    mult=4).to(device)

    model.load_state_dict(torch.load('output/generator.pth')) 
    
    ########################################################################
    #Load Images
    import numpy as np
    from PIL import Image
    
    im_A = Image.open('datasets/facades/train/A/103_A.jpg')
    im_B = Image.open('datasets/facades/train/B/103_B.jpg')

    im_A_arr = np.array(im_A)/256.0
    im_B_arr = np.array(im_B)/256.0
    
    im_A_tensor = torch.tensor(im_A_arr,dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    im_B_tensor = torch.tensor(im_B_arr,dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    
    #def im2tensor(im):
    #    t1 = transforms.ToTensor()
    #    t2 = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    #    return t2(t1(im))
    
    
    
    ########################################################################
    #Forward then Backward
    
    x = im_A_tensor
    #x = torch.tensor(im_arr,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    #x = torch.rand(1, 3, 256, 256) #*2-1.0 #Variable(torch.randn(1, 3, 256, 256))
    print("x.shape = {}".format(x.shape))

    fx = model.forward(x)
    print("fx.shape = {}".format(fx.shape))

    x_rec = model.inverse(fx)
    print("x_rec.shape = {}".format(x_rec.shape))
    print(((x-x_rec)**2).sum())
    
    
    ########################################################################
    #Backward then Forward
    
    y = im_B_tensor
    #y = torch.tensor(im_arr,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    #y = x
    print("y.shape = {}".format(y.shape))
    
    f_inv_y = model.inverse(y)
    print("f_inv_y.shape = {}".format(f_inv_y.shape))
    
    y_rec = model.forward(f_inv_y)
    print("y_rec.shape = {}".format(y_rec.shape))
    print(((y-y_rec)**2).sum())
    
    
    ########################################################################
    #Display 
    import matplotlib.pyplot as plt
    fig,((ax11,ax12,ax13),(ax21,ax22,ax23)) = plt.subplots(2,3,sharex=True,sharey=True)
    
    ax11.imshow(    x[0].permute(1,2,0).cpu())
    ax12.imshow(   fx[0].permute(1,2,0).detach().cpu().numpy())
    ax13.imshow(x_rec[0].permute(1,2,0).detach().cpu().numpy())
    ax11.set_title('x')
    ax12.set_title('f(x)')
    ax13.set_title('f^-1(f(x))')
    
    ax21.imshow(      y[0].permute(1,2,0).cpu())
    ax22.imshow(f_inv_y[0].permute(1,2,0).detach().cpu().numpy())
    ax23.imshow(  y_rec[0].permute(1,2,0).detach().cpu().numpy())
    ax21.set_title('y')
    ax22.set_title('f^-1(y)')
    ax23.set_title('f(f^-1(y))')
    
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig('output/irevnet_facades.png')
    plt.show()
    
    
    ######################################################################## 
    #DONE
