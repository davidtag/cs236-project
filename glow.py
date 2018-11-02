import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transforms import Transform, ComposeTransform

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super(InvertibleConv1x1, self).__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.w = nn.Parameter(torch.Tensor(w_init))

    def forward_w(self):
        return self.w


class InvertibleConv1x1(Transform):
    bijective = True
    event_dim = 3
    def __init__(self, num_channels):
        super(InvertibleConv1x1, self).__init__()
        self.module = nn.Module()
        self._num_channels = num_channels
        w_init = np.linalg.qr(
            np.random.randn(num_channels, num_channels))[0].astype(np.float32)
        self.module.W = nn.Parameter(torch.Tensor(w_init))

    def _call(self, x):
        W = self.module.W.view((self._num_channels, self._num_channels, 1, 1))
        return F.conv2d(x, W)

    def _inverse(self, x):
        W = self.module.W.inverse()
        W = W.view((self._num_channels, self._num_channels, 1, 1))
        return F.conv2d(x, W)

    def log_abs_det_jacobian(self, x, y):
        ndims = np.prod(x.shape[2], x.shape[3])
        return ndims * torch.log(torch.abs(torch.det(self.module.W)))


def split_half(x):
    s = x.shape[1] // 2
    return x[:, :s, ...], x[:, s:, ...]

def padding(k, s):
    return ((k - 1) * s + 1) // 2
    o = [i + 2*p - k - (k-1)*(d-1)]/s + 1


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


if __name__ == "__main__":
    x = torch.rand((32, 10, 32, 32))
    # transform =

    ComposeTransform

    def network():
        return nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 10, kernel_size=3, padding=1),
        )
    transform = ComposeTransform([
        InvertibleConv1x1(10),
        ChannelCoupling(network()),
        InvertibleConv1x1(10),
        ChannelCoupling(network()),
        InvertibleConv1x1(10),
        ChannelCoupling(network()),
        InvertibleConv1x1(10),
        ChannelCoupling(network()),
    ])
    z = transform(x)
    x2 = transform.inv(z)
    # print(np.allclose(x.detach().numpy(), x2.detach().numpy()))
    print(torch.sum(x - x2))
