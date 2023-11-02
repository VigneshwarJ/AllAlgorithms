#pragma once
#include <array>



void addVectorSIMT(float* A, float* B, float* C, size_t size);


class ParallelExamples
{
};


void addVectorSISD(float* A, float* B, float* C, size_t size);

void matrixMultiplySISD(float* A, float* B, float* C, int m, int n, int o);
//void matrixMultiplySISD(float* A, float* B, float* C, int m, int n, int o);

void matrixMultiplySIMTNaive(float* A, float* B, float* C, int m, int n, int o);
void test(int N);