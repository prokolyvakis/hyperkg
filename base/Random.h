/* 
 Copyright (c) 2018-present, the Authors of the OpenKE-PyTorch (old).
 All rights reserved.
 
 Link to the project: https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)
 
 Note: This code was partially adapted by Prodromos Kolyvakis
       to adapt to the case of HyperKG, described in:
       https://arxiv.org/abs/1908.04895
*/

#ifndef RANDOM_H
#define RANDOM_H
#include "Setting.h"
#include <cstdlib>

unsigned long long *next_random;

extern "C"
void randReset() {
	next_random = (unsigned long long *)calloc(workThreads, sizeof(unsigned long long));
	for (INT i = 0; i < workThreads; i++)
		next_random[i] = rand();
}

unsigned long long randd(INT id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

INT rand_max(INT id, INT x) {
	INT res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

//[a,b)
INT rand(INT a, INT b){
	return (rand() % (b-a))+ a;
}
#endif
