# Copyright (c) 2019-present, Prodromos Kolyvakis
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .Model import Model
from torch.autograd import Variable, Function

eps = 1e-5

class Arcosh(Function):
	def __init__(self, eps=eps):
		super(Arcosh, self).__init__()
		self.eps = eps

	def forward(self, x):
		self.z = torch.sqrt(x * x - 1)
		return torch.log(x + self.z)

	def backward(self, g):
		z = torch.clamp(self.z, min=eps)
		z = g / z
		return z

class PoincareDistance(Function):
	boundary = 1. - eps

	def grad(self, x, v, sqnormx, sqnormv, sqdist):
		alpha = (1. - sqnormx)
		beta = (1. - sqnormv)
		z = 1 + 2 * sqdist / (alpha * beta)
		a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
		a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
		z = torch.sqrt(torch.pow(z, 2) - 1)
		z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
		return 4 * a / z.expand_as(x)

	def forward(self, u, v):
		self.save_for_backward(u, v)
		self.squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, self.boundary)
		self.sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, self.boundary)
		self.sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
		x = self.sqdist / ((1. - self.squnorm) * (1. - self.sqvnorm)) * 2 + 1
		# arcosh
		z = torch.sqrt(torch.pow(x, 2) - 1)
		return torch.log(x + z)
		# return 1. / (1. + z + x)

	def backward(self, g):
		u, v = self.saved_tensors
		g = g.unsqueeze(-1)
		gu = self.grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
		gv = self.grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
		return g.expand_as(gu) * gu, g.expand_as(gv) * gv

class Poincare(Model):

	def __init__(self, config):
		super(Poincare,self).__init__(config)
		self.entMaxNorm = 0.5 - 1e-2
		self.relMaxNorm = 1.0 - 1e-2
		self.ent_embeddings=nn.Embedding(config.entTotal,config.hidden_size,
			max_norm=self.entMaxNorm,
            sparse=False,
            scale_grad_by_freq=False)
		self.rel_embeddings=nn.Embedding(config.relTotal,config.hidden_size,
			max_norm=self.relMaxNorm,
            sparse=False,
            scale_grad_by_freq=True)
		self.init_weights()
		# Create Rotation Matrix:
		self.eye = self.cuda_transform(torch.eye(config.hidden_size))
		eye_up, eye_down = torch.chunk(self.eye, 2, 0)
		self.permutation = self.cuda_transform(torch.cat((eye_down, eye_up), 0))
		# Dropout Mask Configuration:
		self.dropout = nn.Dropout(p=0.5, inplace=False)
		
		self.sigmoid = torch.nn.Sigmoid()
		self.softmax = torch.nn.Softmax(dim=-1)
		self.relu = torch.nn.ReLU()
		
	def init_weights(self, scale=1e-4):
		nn.init.xavier_uniform_(self.ent_embeddings.weight.data) # GOLD!
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data) # GOLD!

	
	def _calc(self,h,t,r, func=PoincareDistance):
		t_permuted = t.matmul(self.permutation) # GOLD !
		h_plus_permuted_t = (h + (t_permuted)) # GOLD!
		return func()(r, h_plus_permuted_t) # GOLD!
		
	def _regularizer(self, x, freq_tensor=1., rho=1.):
		return torch.sum(rho - torch.sum(x**2, 1)) # GOLD!
		
	def loss_func(self,p_score,n_score, func='cross_entropy'):
		if func == 'max_margin':
			criterion = self.cuda_transform(nn.MarginRankingLoss(self.config.margin, reduction='sum'))
			y = self.cuda_transform(Variable(torch.Tensor([-1])))
			loss = criterion(p_score,n_score,y)
		elif func == 'hinge_loss': # reduction='sum'
			criterion = self.cuda_transform(nn.HingeEmbeddingLoss(self.config.margin, reduction='sum'))
			p_n_score = torch.cat((p_score, n_score), 1)
			y = (self.cuda_transform(torch.t(self.get_all_labels().view(self.config.negative_ent + self.config.negative_rel + 1, -1)))).type(torch.cuda.LongTensor)
			loss = criterion(p_n_score, y)
		elif func == 'cross_entropy': # reduction='sum'
			criterion = self.cuda_transform(nn.CrossEntropyLoss(reduction='sum'))
			y = self.cuda_transform(torch.zeros(self.config.batch_size).type(torch.cuda.LongTensor))
			p_n_score = -(torch.cat((p_score, n_score), 1))
			loss = criterion(p_n_score, y)
		else:
			raise ValueError('Unknown loss function: {func} in {self.__name__} model.')
		return loss

	def forward(self):
		pos_h,pos_t,pos_r=self.get_postive_instance()
		neg_h,neg_t,neg_r=self.get_negtive_instance()

		p_h=self.ent_embeddings(pos_h) # GOLD!
		p_t=self.ent_embeddings(pos_t) # GOLD!
		p_r=self.rel_embeddings(pos_r) # GOLD!

		n_h=self.ent_embeddings(neg_h) # GOLD!
		n_t=self.ent_embeddings(neg_t) # GOLD!
		n_r=self.rel_embeddings(neg_r) # GOLD!

		_p_score = self._calc(p_h, p_t, p_r)
		_n_score = self._calc(n_h, n_t, n_r)

		p_score = _p_score.view(-1, 1)
		n_score = _n_score.view(-1, self.config.negative_ent + self.config.negative_rel)

		loss = self.loss_func(p_score, n_score, func='max_margin') 

		reg_loss = \
				self._regularizer(p_h) + \
				self._regularizer(p_t) + \
				self._regularizer(p_r) + \
				self._regularizer(n_h) + \
				self._regularizer(n_t) + \
				self._regularizer(n_r)

		return loss + self.config.lmbda * reg_loss

	def predict(self, predict_h, predict_t, predict_r):
		self.eval()
		with torch.no_grad():
			p_h=self.ent_embeddings(self.cuda_transform(Variable(torch.from_numpy(np.asarray(predict_h, dtype=np.int64))))) # GOLD!
			p_t=self.ent_embeddings(self.cuda_transform(Variable(torch.from_numpy(np.asarray(predict_t, dtype=np.int64))))) # GOLD!
			p_r=self.rel_embeddings(self.cuda_transform(Variable(torch.from_numpy(np.asarray(predict_r, dtype=np.int64)))))
			p_score = self._calc(p_h, p_t, p_r)
			return p_score.cpu()
