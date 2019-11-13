#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of the following repo: 
# https://github.com/facebookresearch/poincare-embeddings.
# Note: This code was partially adapted by Prodromos Kolyvakis
#       to adapt to the case of HyperKG, described in:
#       https://arxiv.org/abs/1908.04895
#

import numpy as np
from numpy.random import choice, randint
from torch.utils.data import Dataset
from collections import defaultdict as ddict


def slurp(fname, int_type, sep=' ', reverse=False):
	subs = []
	with open(fname, 'r') as fin:
		lines = fin.readlines()
	ents = dict()
	rels = set()
	for line in lines[1:]:
		h, t, r = map(int, line.strip().split(sep))
		subs.append((h, t, r))
		if h not in ents:
			ents[h] = 1
		else:
			ents[h] += 1
		if t not in ents:
			ents[t] = 0
		if reverse:
			ents[t] += 1
		if r not in rels:
			rels.add(r)
	idx = np.array(subs, dtype=int_type)

	return idx, len(ents.keys()), len(rels), ents

class GraphDataset(Dataset):
	_ntries = 10
	_dampening = 0.75
	_int_type = None

	def __init__(self, idx, nents, nnegs, int_type, unigram_size=1e8, corrupt_both=False):
		print('The PyTorch data loader is being used!')
		self.idx = idx
		self.nnegs = nnegs
		self.burnin = False
		self.nents = nents
		self.max_tries = self.nnegs * self._ntries
		self._int_type = int_type
		self.corrupt_both = corrupt_both

		self._weights = ddict(lambda: ddict(int))
		self._counts = np.ones(self.nents, dtype=np.float32)
		for i in range(idx.shape[0]):
			h, t, _ = self.idx[i]
			self._counts[t] += 1
			self._counts[h] += 1
			self._weights[h][t] += 1.
			self._weights[t][h] += 1.
		self._weights = dict(self._weights)

		if unigram_size > 0:
			c = self._counts ** self._dampening
			self.unigram_table = choice(
				self.nents,
				size=int(unigram_size),
				p=(c / c.sum())
			)

	def __getitem__(self, i):
		h, t, r = self.idx[i]
		negs_h, negs_t = set(), set()
		ntries = 0
		nnegs = self.nnegs
		corrupt_both = self.corrupt_both


		if self.burnin:
			nnegs *= 1#0.1
		while ntries < self.max_tries and len(negs_h) + len(negs_t) < nnegs:
			if self.burnin:
				n = randint(0, len(self.unigram_table))
				n = int(self.unigram_table[n])
			else:
				n = randint(0, self.nents)
			if n not in self._weights[t]:
				if corrupt_both:
					if randint(0,2):
						negs_h.add(n)
					else:
						negs_t.add(n)
				else:
					negs_t.add(n)
			ntries += 1
		if len(negs_t) + len(negs_h) == 0:
			if corrupt_both:
				negs_h.add(h)
				negs_t.add(t)
			else:
				negs_t.add(t)
		negs_h, negs_t = list(negs_h), list(negs_t)
		while len(negs_h) + len(negs_t) < nnegs:
			if corrupt_both:
				if randint(0,2):
					negs_h.append(negs_h[randint(0, len(negs_h))])
				else:
					negs_t.append(negs_t[randint(0, len(negs_t))])
			else:
				negs_t.append(negs_t[randint(0, len(negs_t))])
		if len(negs_h) + len(negs_t) > nnegs:
			if corrupt_both:
				if randint(0,2):
					negs_h.pop(randint(0, len(negs_h)))
				else:
					negs_t.pop(randint(0, len(negs_t)))
			else:
				negs_t.pop(randint(0, len(negs_t)))
		negs_ys = [-1 for _ in range(len(negs_h) + len(negs_t))]
		
		assert len(negs_h) + len(negs_t) <= nnegs, 'A possible error occured in the sampling method!'
		return h, t, r, negs_h, negs_t, 1, negs_ys

	def __len__(self):
		return self.idx.shape[0]

	@classmethod
	def collate(cls, batch):
		hs, ts, rs, ys = [], [], [], []
		negs_hs, negs_ts, negs_rs, negs_ys = [], [], [], []
		for h, t, r, negs_h, negs_t, y, negs_y in batch:
			hs.append(h)
			ts.append(t)
			rs.append(r)
			ys.append(y)
			negs_hs.extend(negs_h)
			negs_ts.extend([t]*(len(negs_h)))
			negs_hs.extend([h]*(len(negs_t)))
			negs_ts.extend(negs_t)
			negs_rs.extend([r]*(len(negs_h)+len(negs_t)))
			negs_ys.extend(negs_y)
		hs.extend(negs_hs)
		ts.extend(negs_ts)
		rs.extend(negs_rs)
		ys.extend(negs_ys)
		
		batch_h = np.array(hs, dtype=cls._int_type)
		batch_t = np.array(ts, dtype=cls._int_type)
		batch_r = np.array(rs, dtype=cls._int_type)
		batch_y = np.array(ys, dtype=cls._int_type)
		
		return batch_h, batch_t, batch_r, batch_y

def load_dataset(fname, nnegs, int_type, sep=' ', unigram_size=1e8):
	idx, nents, _, _ = slurp(fname, int_type, sep)
	return GraphDataset(idx, nents, nnegs, int_type, unigram_size)

def create_adjacencies(fname, nents, int_type, sep=' ', reverse=False):
	idx, _, nrels, ents = slurp(fname, int_type, sep, reverse)

	num_of_connections = [ 
		ents[i]+1 if i in ents else 
		1 for i in range(nents) 
		]
	num_of_connections = np.array(num_of_connections, dtype=int_type)

	data, rows, cols = [], [], []
	for s, o, p in idx.tolist():
		data.append(p)
		rows.append(s)
		cols.append(o)

	rows = rows	 + [i for i in range(nents)]
	cols = cols + [i for i in range(nents)]
	data = data + [nrels for i in range(nents)]

	indices = np.array([rows, cols], dtype=int_type)
	v = np.array(data, dtype=int_type)
	adjacencies = [indices, v, nents]

	return adjacencies, num_of_connections

