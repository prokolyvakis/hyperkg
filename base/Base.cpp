/* 
 Copyright (c) 2018-present, the Authors of the OpenKE-PyTorch (old).
 All rights reserved.
 
 Link to the project: https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)
 
 Note: This code was partially adapted by Prodromos Kolyvakis
       to adapt to the case of HyperKG, described in:
       https://arxiv.org/abs/1908.04895
*/

#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <mutex>
#include <thread>
#include<iostream>
using namespace std;
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <ctime>

extern "C"
void setInPath(char *path);

extern "C"
void setOutPath(char *path);

extern "C"
void setWorkThreads(INT threads);

extern "C"
void setBern(INT con);

extern "C"
INT getWorkThreads();

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

extern "C"
void randReset();

extern "C"
void importTrainFiles();

extern "C"
void getFrequencies(INT *extFreqEnt, INT *extFreqRel);

struct Parameter {
	INT id;
	INT *batch_h;
	INT *batch_t;
	INT *batch_r;
	REAL *batch_y;
	INT batchSize;
	INT negRate;
	INT negRelRate;
};

std::mutex mtx;           // mutex for critical section
INT idx;

INT corrupt_neignbors(INT id) {
	unsigned long long size = (unsigned long long) exRels[id].size();

	if (0 == size)
		return -1; // it should be -1!
	unsigned long long random = rand() % size;
	unsigned long long idx = 0;
	INT value = -1;
	
	for (set<INT>::iterator s = exRels[id].begin(); s != exRels[id].end(); s++, idx++) {
		 value = *s;
		 if (idx == random) break;
	}
	return value;
}

void* getBatch(void* con) {
	
	Parameter *para = (Parameter *)(con);
	INT id = para -> id;
	INT *batch_h = para -> batch_h;
	INT *batch_t = para -> batch_t;
	INT *batch_r = para -> batch_r;
	REAL *batch_y = para -> batch_y;
	INT batchSize = para -> batchSize;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	INT lef, rig;
	if (batchSize % workThreads == 0) {
		lef = id * (batchSize / workThreads);
		rig = (id + 1) * (batchSize / workThreads);
	} else {
		lef = id * (batchSize / workThreads + 1);
		rig = (id + 1) * (batchSize / workThreads + 1);
		if (rig > batchSize) rig = batchSize;
	}
	REAL prob = 500;
	for (INT batch = lef; batch < rig; batch++) {
		INT i = rand_max(id, trainTotal);
		// mtx.lock();
		// if (++idx > trainTotal)
			// idx = 0;
		// INT i = idx;
		// mtx.unlock();
		batch_h[batch] = trainList[i].h;
		batch_t[batch] = trainList[i].t;
		batch_r[batch] = trainList[i].r;
		batch_y[batch] = 1;
		INT last = batchSize;
		for (INT times = 0; times < negRate; times ++) {
			if (bernFlag)
				prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
			if (randd(id) % 1000 < prob) {
				batch_h[batch + last] = trainList[i].h; // GOLD !
				batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r); // GOLD !
				batch_r[batch + last] = trainList[i].r;
			} else {
				// INT corrupted = corrupt_neignbors(trainList[i].t);
				// if (( corrupted == -1 ) || ( corrupted == trainList[i].h ) || ((rand() % 2 == 1)))
				// if (( corrupted == -1 ) || ((rand() % 2 == 1)))
					// corrupted = corrupt_tail(id, trainList[i].t, trainList[i].r);
				// batch_h[batch + last] = corrupted;
				batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r); // GOLD !
				batch_t[batch + last] = trainList[i].t; // GOLD !
				batch_r[batch + last] = trainList[i].r;
			}
			batch_y[batch + last] = -1;
			last += batchSize;
		}
		for (INT times = 0; times < (negRelRate/2); times++) {
			batch_h[batch + last] = trainList[i].h;
			// batch_t[batch + last] = trainList[i].t; // GOLD !
			batch_t[batch + last] =  trainList[i].h;
			// batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t); // GOLD !
			batch_r[batch + last] = trainList[i].r;
			batch_y[batch + last] = -1;
			last += batchSize;
			batch_h[batch + last] = trainList[i].t;
			batch_t[batch + last] =  trainList[i].t;
			// batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t); // GOLD !
			batch_r[batch + last] = trainList[i].r;
			batch_y[batch + last] = -1;
			last += batchSize;
		}
	}
}

extern "C"
void sampling(INT *batch_h, INT *batch_t, INT *batch_r, REAL *batch_y, INT batchSize, INT negRate = 1, INT negRelRate = 0) {
	std::thread *pt = new std::thread[workThreads];
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));
	for (INT threads = 0; threads < workThreads; threads++) {
		para[threads].id = threads;
		para[threads].batch_h = batch_h;
		para[threads].batch_t = batch_t;
		para[threads].batch_r = batch_r;
		para[threads].batch_y = batch_y;
		para[threads].batchSize = batchSize;
		para[threads].negRate = negRate;
		para[threads].negRelRate = negRelRate;
		pt[threads] = std::thread(getBatch, (void*)(para+threads));
	}
	for (INT threads = 0; threads < workThreads; threads++)
		pt[threads].join();
	delete [] pt;
	free(para);
}

int main() {
	importTrainFiles();
	return 0;
}