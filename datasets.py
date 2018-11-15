import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch
import numpy as np
from torch.distributions.transforms import SigmoidTransform

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = img2tensor(np.array(self.transform(\
                                                    Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
                                                   )),batch_dim=False)

        if self.unaligned:
            item_B = img2tensor(np.array(self.transform(\
                          Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB')\
                      )),batch_dim=False)
        else:
            item_B = img2tensor(np.array(self.transform(\
                          Image.open(self.files_B[index % len(self.files_B)]).convert('RGB')\
                      )),batch_dim=False)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    
def img2tensor(im_array,alpha=0.05,permute=True,batch_dim=True):
    """
    @param im_array: np.uint8 array-representation of an image of shape (h,w,c)
    @param alpha: scaling factor to avoid boundary effects
    @param permute: bring channel dimension to the fron of the image
    @param batch_dim: add a batch dimension to the front of the image
    
    @return: torch.float32 tensor of the image on an unbounded domain
             achieved by passing bounded image through a logit (sigmoid^-1 function)
    """
    sigmoid = SigmoidTransform()                                     #initialize sigmoid transformer
    noise   = np.random.uniform(size=im_array.shape)                 #random noise in [0,1)
    im_jittered = torch.tensor(im_array + noise,dtype=torch.float32) #jittered images in [0,256), convert to tensor
    im_norm = alpha+(1-alpha)*im_jittered/256                        #normalized image in (0,1)
    im_tensor = sigmoid.inv(im_norm)                                 #convert to unbounded numbers for modelling
    if permute:
        im_tensor = im_tensor.permute(2,0,1)                         #bring channel dim to front
    if batch_dim:
        im_tensor = im_tensor.unsqueeze(0)                           #add batch dim
    return im_tensor