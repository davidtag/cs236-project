import numpy as np

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

import torch.nn.functional as F
from torch.distributions.transforms import Transform

class RealNVP_Functional2(Transform):
	bijective = True
	def __init__(self):
		super(RealNVP_Functional2, self).__init__()

		nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())
		nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2))
		masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
		prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

		self.prior = prior
		self.mask = nn.Parameter(masks, requires_grad=False)
		self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
		self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

	def _call(self, z):
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

	def _inverse(self,x):
		self.f(x)[0]
		
	def log_abs_det_jacobian(self, z, x):
		return self.f(x)[1]
	
	
class RealNVP(nn.Module):
	def __init__(self):
		super(RealNVP, self).__init__()
		self.transform = RealNVP_Functional2()
		

class RealNVP_Functional(nn.Module):
	def __init__(self):
		super(RealNVP_Functional, self).__init__()

		nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())
		nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2))
		masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
		prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

		self.prior = prior
		self.mask = nn.Parameter(masks, requires_grad=False)
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