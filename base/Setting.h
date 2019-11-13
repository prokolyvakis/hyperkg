/* 
 Copyright (c) 2018-present, the Authors of the OpenKE-PyTorch (old).
 All rights reserved.
 
 Link to the project: https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)
 
 Note: This code was partially adapted by Prodromos Kolyvakis
       to adapt to the case of HyperKG, described in:
       https://arxiv.org/abs/1908.04895
*/

#ifndef SETTING_H
#define SETTING_H
#define INT long
#define REAL float
#include <cstring>
#include <cstdio>
#include <string>
#include <limits>

bool compareREALs(REAL A, REAL B) 
{
   REAL EPSILON = std::numeric_limits<float>::min();
   REAL diff = A - B;
   return (diff < EPSILON) && (-diff < EPSILON);
}

std::string inPath = "../data/FB15K/";
std::string outPath = "../data/FB15K/";

extern "C"
void setInPath(char *path) {
	INT len = strlen(path);
	inPath = "";
	for (INT i = 0; i < len; i++)
		inPath = inPath + path[i];
	printf("Input Files Path : %s\n", inPath.c_str());
}

extern "C"
void setOutPath(char *path) {
	INT len = strlen(path);
	outPath = "";
	for (INT i = 0; i < len; i++)
		outPath = outPath + path[i];
	printf("Output Files Path : %s\n", outPath.c_str());
}

/*
============================================================
*/

INT workThreads = 1;

extern "C"
void setWorkThreads(INT threads) {
	workThreads = threads;
}

extern "C"
INT getWorkThreads() {
	return workThreads;
}

/*
============================================================
*/

INT relationTotal = 0;
INT entityTotal = 0;
INT tripleTotal = 0;
INT testTotal = 0;
INT trainTotal = 0;
INT validTotal = 0;

extern "C"
INT getEntityTotal() {
	return entityTotal;
}

extern "C"
INT getRelationTotal() {
	return relationTotal;
}

extern "C"
INT getTripleTotal() {
	return tripleTotal;
}

extern "C"
INT getTrainTotal() {
	return trainTotal;
}

extern "C"
INT getTestTotal() {
	return testTotal;
}

extern "C"
INT getValidTotal() {
	return validTotal;
}
/*
============================================================
*/

INT bernFlag = 0;

extern "C"
void setBern(INT con) {
	bernFlag = con;
}

#endif
