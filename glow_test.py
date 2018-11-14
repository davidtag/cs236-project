import sys

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

from glow import Glow

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

def draw_image_row(tensors,labels,save_name=None):
    assert(len(tensors)==len(labels))
    n = len(tensors)
    fig,ax = plt.subplots(1,n,figsize=(4*n,4),sharex=True,sharey=True)
    for a,t,l in zip(ax,tensors,labels):
        a.imshow(tensor2img(t))
        a.set_title(l)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    if save_name is not None:
        plt.savefig(save_name)
    #plt.show()

generator = Glow(16, 3, 32, squeeze=4).cuda()
generator.load_state_dict(torch.load('output/glow/{}-{}-generator.pth'.format(sys.argv[1],sys.argv[2]))) 

A = img2tensor(np.array(Image.open('datasets/facades/train/A/103_A.jpg'))).cuda()
B = img2tensor(np.array(Image.open('datasets/facades/train/B/103_B.jpg'))).cuda()


draw_image_row([A,generator.transform(A),generator.transform.inv(A)],['A','f(A)','f^-1(A)'],'output/glow/facades_img2mask.png')
draw_image_row([B,generator.transform.inv(B),generator.transform(B)],['B','g(B)','g^-1(B)'],'output/glow/facades_mask2img.png')