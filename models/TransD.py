# 
#  Copyright (c) 2018-present, the Authors of the OpenKE-PyTorch (old).
#  All rights reserved.
#
#  Link to the project: https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)
#

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model

class TransD(Model):
	def __init__(self,config):
		super(TransD,self).__init__(config)
		self.ent_embeddings=nn.Embedding(self.config.entTotal,self.config.hidden_size)
		self.rel_embeddings=nn.Embedding(self.config.relTotal,self.config.hidden_size)
		self.ent_transfer=nn.Embedding(self.config.entTotal,self.config.hidden_size)
		self.rel_transfer=nn.Embedding(self.config.relTotal,self.config.hidden_size)
		self.init_weights()
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_embeddings.weight.data)
		nn.init.xavier_uniform(self.ent_transfer.weight.data)
		nn.init.xavier_uniform(self.rel_transfer.weight.data)
	r'''
	TransD constructs a dynamic mapping matrix for each entity-relation pair by 
	considering the diversity of entities and relations simultaneously. 
	Compared with TransR/CTransR, TransD has fewer parameters and 
	has no matrix vector multiplication.
	'''
	def _transfer(self,e,t,r):
		return F.normalize(e+torch.sum(e*t,1,True)*r,2,1)
	def _calc(self,h,t,r):
		return torch.abs(h+r-t)
	def loss_func(self,p_score,n_score):
		criterion= nn.MarginRankingLoss(self.config.margin,False).cuda()
		y=Variable(torch.Tensor([-1])).cuda()
		loss=criterion(p_score,n_score,y)
		return loss
	def forward(self):
		pos_h,pos_t,pos_r=self.get_postive_instance()
		neg_h,neg_t,neg_r=self.get_negtive_instance()
		p_h_e=self.ent_embeddings(pos_h)
		p_t_e=self.ent_embeddings(pos_t)
		p_r_e=self.rel_embeddings(pos_r)
		n_h_e=self.ent_embeddings(neg_h)
		n_t_e=self.ent_embeddings(neg_t)
		n_r_e=self.rel_embeddings(neg_r)
		p_h_t=self.ent_transfer(pos_h)
		p_t_t=self.ent_transfer(pos_t)
		p_r_t=self.rel_transfer(pos_r)
		n_h_t=self.ent_transfer(neg_h)
		n_t_t=self.ent_transfer(neg_t)
		n_r_t=self.rel_transfer(neg_r)
		p_h=self._transfer(p_h_e,p_h_t,p_r_t)
		p_t=self._transfer(p_t_e,p_t_t,p_r_t)
		p_r=p_r_e
		n_h=self._transfer(n_h_e,n_h_t,n_r_t)
		n_t=self._transfer(n_t_e,n_t_t,n_r_t)
		n_r=n_r_e
		_p_score = self._calc(p_h, p_t, p_r)
		_n_score = self._calc(n_h, n_t, n_r)
		_p_score = _p_score.view(-1, 1, self.config.hidden_size)
		_n_score = _n_score.view(-1, self.config.negative_ent + self.config.negative_rel, self.config.hidden_size)
		p_score=torch.sum(torch.mean(_p_score, 1),1)
		n_score=torch.sum(torch.mean(_n_score, 1),1)
		loss=self.loss_func(p_score,n_score)
		return loss
	def predict(self, predict_h, predict_t, predict_r):
		p_h_e=self.ent_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
		p_t_e=self.ent_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
		p_r_e=self.rel_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
		p_h_t=self.ent_transfer(Variable(torch.from_numpy(predict_h)).cuda())
		p_t_t=self.ent_transfer(Variable(torch.from_numpy(predict_t)).cuda())
		p_r_t=self.rel_transfer(Variable(torch.from_numpy(predict_r)).cuda())
		p_h=self._transfer(p_h_e,p_h_t,p_r_t)
		p_t=self._transfer(p_t_e,p_t_t,p_r_t)
		p_r=p_r_e
		_p_score = self._calc(p_h, p_t, p_r)
		p_score=torch.sum(_p_score,1)
		return p_score.cpu()
