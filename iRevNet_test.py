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

from models import iRevNetCycle
           

if __name__ == '__main__':
    ########################################################################
    #CODE TO TEST IMPLEMENTATION
    
    #import pdb
    #pdb.set_trace()
    device = torch.device('cuda')
    
    ########################################################################
    #Initialize Model
    #model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2,2,2,2],
    #                nChannels=[24,96,384,1536], nClasses=1000, init_ds=2,
    #                dropout_rate=0., affineBN=True, in_shape=[3, 256, 256],
    #                mult=4).to(device)

    model = iRevNetCycle([3,256,256]).to(device)
    model.load_state_dict(torch.load('output/generator.pth')) 
    
    ########################################################################
    #Load Images
    import numpy as np
    from PIL import Image
    
    im_A = Image.open('datasets/horse2zebra/train/A/n02381460_1001.jpg') #('datasets/facades/train/A/103_A.jpg')
    im_B = Image.open('datasets/horse2zebra/train/B/n02391049_10007.jpg') #('datasets/facades/train/B/103_B.jpg')

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

    fx = model.transform(x)
    print("fx.shape = {}".format(fx.shape))

    x_rec = model.transform.inv(fx)
    print("x_rec.shape = {}".format(x_rec.shape))
    print(((x-x_rec)**2).sum())
    
    
    ########################################################################
    #Backward then Forward
    
    y = im_B_tensor
    #y = torch.tensor(im_arr,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    #y = x
    print("y.shape = {}".format(y.shape))
    
    f_inv_y = model.transform.inv(y)
    print("f_inv_y.shape = {}".format(f_inv_y.shape))
    
    y_rec = model.transform(f_inv_y)
    print("y_rec.shape = {}".format(y_rec.shape))
    print(((y-y_rec)**2).sum())
    
    
    ########################################################################
    #Display 
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,((ax11,ax12,ax13),(ax21,ax22,ax23)) = plt.subplots(2,3,figsize=(15,10),sharex=True,sharey=True)
    
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
    #plt.show()
    
    
    ######################################################################## 
    #DONE
