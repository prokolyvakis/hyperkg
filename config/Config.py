#coding:utf-8
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
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from .rsgd import RiemannianSGD, euclidean_retraction, poincare_grad
from .data import create_adjacencies
import torch.optim as optim
import os
import time
import datetime
import ctypes
import json

class Config(object):
	r'''
	use ctypes to call C functions from python and set essential parameters.
	'''
	def __init__(self):
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../release/Base.so'))		
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
		self.lib.getFrequencies.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64]
		self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64]
		self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.test_flag = False
		self.in_path = "./"
		self.out_path = "./"
		self.bern = 0
		self.hidden_size = 100
		self.ent_size = self.hidden_size
		self.rel_size = self.hidden_size
		self.train_times = 0
		self.margin = 1.0
		self.nbatches = 100
		self.negative_ent = 1
		self.negative_rel = 0
		self.workThreads = 1
		self.alpha = 0.001
		self.lr_multiplier = 1.0
		self.burn_in_epochs = 30
		self.lmbda = 0.000
		self.gpu_activated = False
		self.log_on = 1
		self.lr_decay=0.000
		self.weight_decay=0.000
		self.exportName = None
		self.importName = None
		self.export_steps = 0
		self.opt_method = "SGD"
		self.optimizer = None
		self.test_link_prediction = False
		self.test_triple_classification = False
		self.valid_every = 5
		self.int_type = np.int32
		self.data_loader_on = False
		self.train_data = None
		self.data_loader = None
		self.dataloader_iterator = None
		self.data_adjacencies = None
		
	def init_link_prediction(self):
		r'''
		import essential files and set essential interfaces for link prediction
		'''
		self.lib.importTestFiles()
		self.lib.importTypeFiles()
		self.test_h = np.zeros(self.lib.getEntityTotal(), dtype = self.int_type)	
		self.test_t = np.zeros(self.lib.getEntityTotal(), dtype = self.int_type)
		self.test_r = np.zeros(self.lib.getEntityTotal(), dtype = self.int_type)
		self.test_h_addr = self.test_h.__array_interface__['data'][0]
		self.test_t_addr = self.test_t.__array_interface__['data'][0]
		self.test_r_addr = self.test_r.__array_interface__['data'][0]

	def init_triple_classification(self):
		r'''
		import essential files and set essential interfaces for triple classification
		'''
		self.lib.importTestFiles()
		self.lib.importTypeFiles()
		self.test_pos_h = np.zeros(self.lib.getTestTotal(), dtype = self.int_type)
		self.test_pos_t = np.zeros(self.lib.getTestTotal(), dtype = self.int_type)
		self.test_pos_r = np.zeros(self.lib.getTestTotal(), dtype = self.int_type)
		self.test_neg_h = np.zeros(self.lib.getTestTotal(), dtype = self.int_type)
		self.test_neg_t = np.zeros(self.lib.getTestTotal(), dtype = self.int_type)
		self.test_neg_r = np.zeros(self.lib.getTestTotal(), dtype = self.int_type)
		self.test_pos_h_addr = self.test_pos_h.__array_interface__['data'][0]
		self.test_pos_t_addr = self.test_pos_t.__array_interface__['data'][0]
		self.test_pos_r_addr = self.test_pos_r.__array_interface__['data'][0]
		self.test_neg_h_addr = self.test_neg_h.__array_interface__['data'][0]
		self.test_neg_t_addr = self.test_neg_t.__array_interface__['data'][0]
		self.test_neg_r_addr = self.test_neg_r.__array_interface__['data'][0]
		self.valid_pos_h = np.zeros(self.lib.getValidTotal(), dtype = self.int_type)
		self.valid_pos_t = np.zeros(self.lib.getValidTotal(), dtype = self.int_type)
		self.valid_pos_r = np.zeros(self.lib.getValidTotal(), dtype = self.int_type)
		self.valid_neg_h = np.zeros(self.lib.getValidTotal(), dtype = self.int_type)
		self.valid_neg_t = np.zeros(self.lib.getValidTotal(), dtype = self.int_type)
		self.valid_neg_r = np.zeros(self.lib.getValidTotal(), dtype = self.int_type)
		self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
		self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
		self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
		self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
		self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
		self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]
		self.relThresh = np.zeros(self.lib.getRelationTotal(), dtype = np.float32)
		self.relThresh_addr = self.relThresh.__array_interface__['data'][0]

	# prepare for train and test
	def init(self):
		self.trainModel = None
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
			self.lib.setBern(self.bern)
			self.lib.setWorkThreads(self.workThreads)
			self.lib.randReset()
			self.lib.importTrainFiles()
			self.relTotal = self.lib.getRelationTotal()
			self.entTotal = self.lib.getEntityTotal()
			self.trainTotal = self.lib.getTrainTotal()
			self.testTotal = self.lib.getTestTotal()
			self.validTotal = self.lib.getValidTotal()
			self.batch_size = int(self.lib.getTrainTotal() / self.nbatches)
			self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
			self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = self.int_type)
			self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = self.int_type)
			self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = self.int_type)
			self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
			self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
			self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
			self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
			self.batch_y_addr = self.batch_y.__array_interface__['data'][0]
			self.check_for_data_loader()
			self.data_adjacencies = create_adjacencies(self.in_path + 'train2id.txt', self.entTotal, int_type=self.int_type, reverse=True)
		if self.test_link_prediction:
			self.init_link_prediction()
		if self.test_triple_classification:
			self.init_triple_classification()

	def get_ent_total(self):
		return self.entTotal

	def get_rel_total(self):
		return self.relTotal

	def set_lmbda(self, lmbda):
		self.lmbda = lmbda

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_opt_method(self, method):
		self.opt_method = method

	def set_test_link_prediction(self, flag):
		self.test_link_prediction = flag

	def set_test_triple_classification(self, flag):
		self.test_triple_classification = flag

	def set_log_on(self, flag):
		self.log_on = flag

	def set_data_loader(self, flag):
		self.data_loader_on = flag
		
	def set_gpu(self, flag):
		self.gpu_activated = flag and torch.cuda.is_available()

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_in_path(self, path):
		self.in_path = path

	def set_out_files(self, path):
		self.out_path = path

	def set_bern(self, bern):
		self.bern = bern

	def set_dimension(self, dim):
		self.hidden_size = dim
		self.ent_size = dim
		self.rel_size = dim

	def set_ent_dimension(self, dim):
		self.ent_size = dim

	def set_rel_dimension(self, dim):
		self.rel_size = dim

	def set_train_times(self, times):
		self.train_times = times

	def set_valid_every(self, times):
		self.valid_every = times

	def set_burn_in_epochs(self, times):
		self.burn_in_epochs = times

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches
	
	def set_margin(self, margin):
		self.margin = margin
	
	def set_work_threads(self, threads):
		self.workThreads = threads
	
	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate
	
	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_int_type(self, t='int32'):
		if t == 'int32':
			self.int_type = np.int32
		elif t == 'int64':
			self.int_type = np.int64
		else:
			raise ValueError('Not a proper integer type: {t}.')
	
	def set_import_files(self, path):
		self.importName = path
	
	def set_export_files(self, path):
		self.exportName = path
	
	def set_export_steps(self, steps):
		self.export_steps = steps
	
	def set_lr_decay(self,lr_decay):
		self.lr_decay=lr_decay
	
	def set_weight_decay(self,weight_decay):
		self.weight_decay=weight_decay

	def belongs_in_poincare_family(self):
		return type(self.trainModel).__name__ in ['Poincare']

	def check_for_data_loader(self):
		if self.data_loader_on:
			from .data import load_dataset
			from torch.utils.data import DataLoader
			self.train_data = load_dataset(self.in_path + 'train2id.txt', nnegs=self.negative_ent, int_type=self.int_type)
			# self.data_loader = DataLoader(self.train_data, batch_size=self.batch_size ,shuffle=True, num_workers=self.workThreads, collate_fn=self.train_data.collate)
			self.data_loader = DataLoader(self.train_data, batch_size=self.batch_size ,shuffle=True, num_workers=0, collate_fn=self.train_data.collate)

	# call function for sampling
	def sampling(self):
		if self.data_loader_on:
			# call pytorch function for sampling
			# print('The pytorch sampling is being used!')
			batch_h, batch_t, batch_r, batch_y = next(self.dataloader_iterator)
			batch_size = batch_h.shape[0]
			self.batch_h[:batch_size], self.batch_t[:batch_size], self.batch_r[:batch_size], self.batch_y[:batch_size] = batch_h, batch_t, batch_r, batch_y
		else:
			# call c function for sampling
			# print('The c custom sampling is being used!')
			self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)

	# save model
	def save_pytorch(self):
		torch.save(self.trainModel.state_dict(), self.exportName)
	# restore model
	def restore_pytorch(self):
		self.trainModel.load_state_dict(torch.load(self.importName))
	
	# save model
	def export_variables(self, path = None):
		if path == None:
			torch.save(self.trainModel.state_dict(), self.exportName)
		else:
			torch.save(self.trainModel.state_dict(), path)

	def import_variables(self, path = None):
		if path == None:
			self.trainModel.load_state_dict(torch.load(self.importName))
		else:
			self.trainModel.load_state_dict(torch.load(path))

	def get_parameter_lists(self):
		return self.trainModel.cpu().state_dict()

	def get_parameters_by_name(self, var_name):
		return self.trainModel.cpu().state_dict().get(var_name)
	# return dict of parameters
	# parameter_name -> parameters
	def get_parameters(self, mode = "numpy"):
		res = {}
		lists = self.get_parameter_lists()
		for var_name in lists:
			if mode == "numpy":
				res[var_name] = lists[var_name].numpy()
			if mode == "list":
				res[var_name] = lists[var_name].numpy().tolist()
			else:
				res[var_name] = lists[var_name]
		return res

	def save_parameters(self, path = None):
		if path == None:
			path = self.out_path
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def set_parameters_by_name(self, var_name, tensor):
		self.trainModel.state_dict().get(var_name).copy_(torch.from_numpy(np.array(tensor)))

	def set_parameters(self, lists):
		for i in lists:
			self.set_parameters_by_name(i, lists[i])

	def set_model(self, model):
		self.model = model
		self.trainModel = self.model(config = self)
		if self.gpu_activated:
			self.trainModel.cuda()
		if self.optimizer != None:
			pass
		elif self.opt_method == "RiemannianSGD" or self.opt_method == "RSGD":
			self.optimizer = RiemannianSGD(self.trainModel.parameters(),rgrad=poincare_grad,retraction=euclidean_retraction,lr=self.alpha)
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr=self.alpha,lr_decay=self.lr_decay,weight_decay=self.weight_decay)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr=self.alpha)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(self.trainModel.parameters(), lr=self.alpha)
		else:
			self.optimizer = optim.SGD(self.trainModel.parameters(), lr=self.alpha)

	def run(self):
		from torch.optim.lr_scheduler import ExponentialLR
		torch.manual_seed(0)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		np.random.seed(0)
		if self.importName != None:
			self.restore_pytorch()
		for epoch in range(self.train_times):
			res = 0.0
			lr = self.alpha
			# lr = self.alpha * (0.9 ** (epoch // 100))
			if self.data_loader_on:
				self.train_data.burnin = False
				self.dataloader_iterator = iter(self.data_loader)
			if self.belongs_in_poincare_family() and (epoch + 1) <= self.burn_in_epochs:
				# self.data_loader_on = False
				if self.data_loader_on:
					self.train_data.burnin = True
				lr = self.lr_multiplier * self.alpha
				
			self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
			self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = self.int_type)
			self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = self.int_type)
			self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = self.int_type)
			self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
			self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
			self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
			self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
			self.batch_y_addr = self.batch_y.__array_interface__['data'][0]
			for batch in range(self.nbatches):
				self.sampling()
				self.optimizer.zero_grad()
				loss = self.trainModel()
				res = res + loss.item()
				loss.backward()
				if self.opt_method == "RiemannianSGD" or self.opt_method == "RSGD":
					self.optimizer.step(lr=lr)
				else:
					self.optimizer.step()
			if self.exportName != None and (self.export_steps!=0 and epoch % self.export_steps == 0):
				self.save_pytorch()
			if self.log_on == 1:
				print(f'Epoch {epoch}: loss: {res}')
			if (epoch+1) % self.valid_every == 0:
				print(f'Validation begins.')
				self.test(epoch+1, 0)
		if self.exportName != None:
			self.save_pytorch()
		if self.out_path != None:
			self.save_parameters(self.out_path)

	def test(self, save_epoch, show=0):
		self.lib.zeroOut()
		if self.importName != None:
			self.restore_pytorch()
		if self.test_link_prediction:
			total = self.lib.getTestTotal()
			for epoch in range(total):
				self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
				res = self.trainModel.predict(self.test_h, self.test_t, self.test_r)
				self.lib.testHead(res.data.numpy().__array_interface__['data'][0], show)
				if epoch % 1000 == 0:
					np.savetxt("./debug/head_res_" + str(epoch) + ".txt", res.detach().numpy(), newline=" ")
				self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
				res = self.trainModel.predict(self.test_h, self.test_t, self.test_r)
				self.lib.testTail(res.data.numpy().__array_interface__['data'][0], show)
				if epoch % 1000 == 0:
					np.savetxt("./debug/tail_res_" + str(epoch) + ".txt", res.detach().numpy(), newline=" ")
				if self.log_on and show == 1:
					print(epoch)
			self.lib.test_link_prediction()
			save_path = './debug/' + self.in_path.split('/')[-2] + '/hyperkg_' + str(save_epoch)
			self.export_variables(save_path + '.pt')
			# self.save_parameters(save_path + '.vec.json')
		if self.test_triple_classification:
			self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
			res_pos = self.trainModel.predict(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
			res_neg = self.trainModel.predict(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
			self.lib.getBestThreshold(self.relThresh_addr, res_pos.data.numpy().__array_interface__['data'][0], res_neg.data.numpy().__array_interface__['data'][0])
			self.lib.getTestBatch(self.test_pos_h_addr, self.test_pos_t_addr, self.test_pos_r_addr, self.test_neg_h_addr, self.test_neg_t_addr, self.test_neg_r_addr)

			res_pos = self.trainModel.predict(self.test_pos_h, self.test_pos_t, self.test_pos_r)
			res_neg = self.trainModel.predict(self.test_neg_h, self.test_neg_t, self.test_neg_r)
			self.lib.test_triple_classification(self.relThresh_addr, res_pos.data.numpy().__array_interface__['data'][0], res_neg.data.numpy().__array_interface__['data'][0])

	def predict_head_entity(self, t, r, k):
		r'''This mothod predicts the top k head entities given tail entity and relation.
		
		Args: 
			t (int): tail entity id
			r (int): relation id
			k (int): top k head entities
		
		Returns:
			list: k possible head entity ids 	  	
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_pytorch()
		test_h = np.array(range(self.entTotal))
		test_r = np.array([r] * self.entTotal)
		test_t = np.array([t] * self.entTotal)
		res = self.trainModel.predict(test_h, test_t, test_r).data.numpy().reshape(-1).argsort()[:k]
		print(res)
		return res
	
	def predict_tail_entity(self, h, r, k):
		r'''This method predicts the tail entities given head entity and relation.
		
		Args：
			h (int): head entity id
			r (int): relation id
			k (int): top k tail entities
		
		Returns:
			list: k possible tail entity ids
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_pytorch()
		test_h = np.array([h] * self.entTotal)
		test_r = np.array([r] * self.entTotal)
		test_t = np.array(range(self.entTotal))
		res = self.trainModel.predict(test_h, test_t, test_r).data.numpy().reshape(-1).argsort()[:k]
		print(res)
		return res

	def predict_relation(self, h, t, k):
		r'''This methods predict the relation id given head entity and tail entity.
		
		Args:
			h (int): head entity id
			t (int): tail entity id
			k (int): top k relations
		
		Returns:
			list: k possible relation ids
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_pytorch()
		test_h = np.array([h] * self.relTotal)
		test_r = np.array(range(self.relTotal))
		test_t = np.array([t] * self.relTotal)
		res = self.trainModel.predict(test_h, test_t, test_r).data.numpy().reshape(-1).argsort()[:k]
		print(res)
		return res

	def predict_triple(self, h, t, r, thresh = None):
		r'''This method tells you whether the given triple (h, t, r) is correct of wrong
	
		Args:
			h (int): head entity id
			t (int): tail entity id
			r (int): relation id
			thresh (fload): threshold for the triple
		'''
		self.init_triple_classification()
		if self.importName != None:
			self.restore_pytorch()	
		res = self.trainModel.predict(np.array([h]), np.array([t]), np.array([r])).data.numpy()
		if thresh != None:
			if res < thresh:
				print("triple (%d,%d,%d) is correct" % (h, t, r))
			else:
				print("triple (%d,%d,%d) is wrong" % (h, t, r))	
			return
		self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
		res_pos = self.trainModel.predict(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
		res_neg = self.trainModel.predict(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
		self.lib.getBestThreshold(self.relThresh_addr, res_pos.data.numpy().__array_interface__['data'][0], res_neg.data.numpy().__array_interface__['data'][0])
		if res < self.relThresh[r]:
			print("triple (%d,%d,%d) is correct" % (h, t, r))
		else: 
			print("triple (%d,%d,%d) is wrong" % (h, t, r))
