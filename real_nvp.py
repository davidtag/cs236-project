import numpy as np

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

import torch.nn.functional as F
from torch.distributions.transforms import Transform


#######################################################################################
class Squeeze(transforms.Transform):
    bijective = True
    event_dim = 3
    def __init__(self, factor):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(factor, int)
        self._factor = factor

    def _call(self, input):
        factor = self._factor
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)
        return x

    def _inverse(self, input):
        factor = self._factor
        factor2 = factor ** 2
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x

    def log_abs_det_jacobian(self, x, y):
        return 0

def split_half(x):
    s = x.shape[1] // 2
    return x[:, :s, ...], x[:, s:, ...]

class ChannelCoupling(transforms.Transform):
    bijective = True
    event_dim = 3

    def __init__(self, network):
        super(ChannelCoupling, self).__init__()
        self._nn = network

    def _call(self, x):
        x1, x2 = split_half(x)
        offset, scale = split_half(self._nn(x1))
        scale = torch.sigmoid(scale + 2.)
        y2 = x2 * scale + offset
        y = torch.cat([x1, y2], dim=1)
        return y

    def _inverse(self, x):
        x1, x2 = split_half(x)
        offset, scale = split_half(self._nn(x1))
        scale = torch.sigmoid(scale + 2.)
        y2 = (x2 - offset) / scale
        y = torch.cat([x1, y2], dim=1)
        return y

    def log_abs_det_jacobian(self, x, y):
        x1, x2 = split_half(x)
        _, scale = split_half(self._nn(x1))
        return torch.sum(torch.log(scale), dim=[1, 2, 3])

def _padding(k, s):
    return ((k - 1) * s + 1) // 2

def _conv(in_dim, out_dim, kernel_size, zero=False):
    conv = nn.Conv2d(in_dim,
                     out_dim,
                     kernel_size=kernel_size,
                     padding=_padding(kernel_size, 1))
    if zero:
        nn.init.constant_(conv.weight, 0.)
        nn.init.constant_(conv.bias, 0.)
    return conv

def TanhTransform():
    return transforms.ComposeTransform(
        [transforms.AffineTransform(loc=0., scale=2.),
         transforms.SigmoidTransform(),
         transforms.AffineTransform(loc=-1., scale=2.)])
		 
		 
#######################################################################################
class RealNVP(nn.module):
	def __init__(self):
		super(RealNVP, self).__init__()
		
		#Prior
		self.prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
		
		#Mask
		mask = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
		self.mask = nn.Parameter(mask, requires_grad=False)
		
		#Trainable Networks
		self.s = torch.nn.ModuleList([self._net_s() for _ in range(len(mask))])
		self.t = torch.nn.ModuleList([self._net_t() for _ in range(len(mask))])
		
    def _coupling_network(self):
        c, h = self._c, self._h
        module = nn.Sequential(
            _conv(c//2, h, kernel_size=7),
            nn.InstanceNorm2d(h),
            nn.ReLU(inplace=False),
            _conv(h, h, kernel_size=3),
            nn.InstanceNorm2d(h),
            nn.ReLU(inplace=False),
            _conv(h, c, kernel_size=7, zero=True),
        )
        self.add_module("couple_{}".format(self._n_couple), module)
        self._n_couple += 1
        return module
		
		
		


#######################################################################################
#######################################################################################
#######################################################################################	

class RealNVP_Functional(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x