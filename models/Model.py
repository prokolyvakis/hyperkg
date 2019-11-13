# 
#  Copyright (c) 2018-present, the Authors of the OpenKE-PyTorch (old).
#  All rights reserved.
#
#  Link to the project: https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)
#
#  Note: This code was partially adapted by Prodromos Kolyvakis
#        to adapt to the case of HyperKG, described in:
#        https://arxiv.org/abs/1908.04895
#

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class Model(nn.Module):

	def __init__(self,config):
		super(Model,self).__init__()
		self.config = config
		self.gpu_activated = config.gpu_activated
		
	def cuda_transform(self,variable):
		if self.gpu_activated:
			return variable.cuda()
		return variable

	def get_postive_instance(self):
		self.postive_h = Variable(torch.from_numpy(np.asarray(self.config.batch_h[0:self.config.batch_size], dtype=np.int64)))
		self.postive_t = Variable(torch.from_numpy(np.asarray(self.config.batch_t[0:self.config.batch_size], dtype=np.int64)))
		self.postive_r = Variable(torch.from_numpy(np.asarray(self.config.batch_r[0:self.config.batch_size], dtype=np.int64)))
		return self.cuda_transform(self.postive_h),self.cuda_transform(self.postive_t),self.cuda_transform(self.postive_r)

	def get_negtive_instance(self):
		self.negtive_h = Variable(torch.from_numpy(np.asarray(self.config.batch_h[self.config.batch_size:self.config.batch_seq_size], dtype=np.int64)))
		self.negtive_t = Variable(torch.from_numpy(np.asarray(self.config.batch_t[self.config.batch_size:self.config.batch_seq_size], dtype=np.int64)))
		self.negtive_r = Variable(torch.from_numpy(np.asarray(self.config.batch_r[self.config.batch_size:self.config.batch_seq_size], dtype=np.int64)))
		return self.cuda_transform(self.negtive_h),self.cuda_transform(self.negtive_t),self.cuda_transform(self.negtive_r)

	def get_all_instance(self):
		self.batch_h = Variable(torch.from_numpy(np.asarray(self.config.batch_h, dtype=np.int64)))
		self.batch_t = Variable(torch.from_numpy(np.asarray(self.config.batch_t, dtype=np.int64)))
		self.batch_r = Variable(torch.from_numpy(np.asarray(self.config.batch_r, dtype=np.int64)))
		return self.cuda_transform(self.batch_h), self.cuda_transform(self.batch_t), self.cuda_transform(self.batch_r)

	def get_all_labels(self):
		self.batch_y=Variable(torch.from_numpy(np.asarray(self.config.batch_y, dtype=np.int64)))
		return self.cuda_transform(self.batch_y)

	def get_adjacencies(self):
		[indices, v, nents] = self.config.data_adjacencies[0]
		indices = Variable(torch.from_numpy(np.asarray(indices, dtype=np.int64)))
		indices = (self.cuda_transform(indices)).detach()
		v = Variable(torch.from_numpy(np.asarray(v, dtype=np.int64)))
		v = (self.cuda_transform(v)).detach()

		deg = self.config.data_adjacencies[1]
		deg = Variable(torch.from_numpy(np.asarray(deg, dtype=np.float32)))
		deg = (self.cuda_transform(deg)).detach()
		return ([indices, v, nents], deg)


	def predict(self):
		pass

	def forward(self):
		pass

	def loss_func(self):
		pass
