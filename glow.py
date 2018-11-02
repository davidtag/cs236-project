import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transforms import Transform, ComposeTransform


class Squeeze(Transform):
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



class InvertibleConv1x1(Transform):
    bijective = True
    event_dim = 3
    def __init__(self, module):
        super(InvertibleConv1x1, self).__init__()
        self.module = module

    def _call(self, x):
        W = self.module.forward_kernel()
        return F.conv2d(x, W)

    def _inverse(self, x):
        W = self.module.inverse_kernel()
        return F.conv2d(x, W)

    def log_abs_det_jacobian(self, x, y):
        ndims = np.prod(x.shape[2], x.shape[3])
        return ndims * torch.log(torch.abs(torch.det(self.module.W)))


def split_half(x):
    s = x.shape[1] // 2
    return x[:, :s, ...], x[:, s:, ...]


class ChannelCoupling(Transform):
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



class InvertibleConv1x1Matrix(nn.Module):
    def __init__(self, c):
        super(InvertibleConv1x1Matrix, self).__init__()
        self._c = c
        w_init = np.linalg.qr(
            np.random.randn(self._c, self._c))[0].astype(np.float32)
        self.W = nn.Parameter(torch.Tensor(w_init))

    def forward_kernel(self):
        return self.W.view((self._c, self._c, 1, 1))

    def inverse_kernel(self):
        return self.W.inverse().view((self._c, self._c, 1, 1))


class Glow(nn.Module):
    def __init__(self, n, c, h, squeeze_factor=2):
        super(Glow, self).__init__()
        self._n_couple = 0
        self._n_conv1x1 = 0
        self._n, self._c, self._h = n, c, h

        transforms = []
        if squeeze_factor > 1:
            transforms.append(Squeeze(squeeze_factor))
            self._c = self._c * (squeeze_factor ** 2)
        for _ in range(self._n):
            transforms.extend([
                InvertibleConv1x1(self._conv1x1_network()),
                ChannelCoupling(self._coupling_network())
            ])
        self.transform = ComposeTransform(transforms)

    def _coupling_network(self):
        c, h = self._c, self._h
        module = nn.Sequential(
            _conv(c//2, h, kernel_size=3),
            nn.ReLU(inplace=False),
            _conv(h, h, kernel_size=1),
            nn.ReLU(inplace=False),
            _conv(h, c, kernel_size=3, zero=True),
        )
        self.add_module("couple_{}".format(self._n_couple), module)
        self._n_couple += 1
        return module

    def _conv1x1_network(self):
        module = InvertibleConv1x1Matrix(self._c)
        self.add_module("conv1x1_{}".format(self._n_conv1x1), module)
        self._n_conv1x1 += 1
        return module


if __name__ == "__main__":
    x = torch.rand((32, 3, 32, 32))

    glow = Glow(10, 3, 64)
    z = glow.transform(x)
    x2 = glow.transform.inv(z)
    print(torch.sum(x - x2))
    print(list(glow.children()))
