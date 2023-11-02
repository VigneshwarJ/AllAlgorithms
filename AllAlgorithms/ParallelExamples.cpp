#include "ParallelExamples.h"
#include <vector>
#include <chrono>
#include <iostream>

void addVectorSISD(float* A, float* B, float* C, size_t size)
{

	for (size_t i = 0; i < size; i++)
	{
		C[i] = (A[i] + B[i]);
	}
}

void matrixMultiplySISD(float* A, float* B, 
	float* C, int m, int n, int o)
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < o; j++)
		{
			C[i * m + j] = 0;
			for (size_t k = 0; k < n; k++)
			{
				C[i*m+ j] += A[i*m+k] * B[k*n+j];
			}
		}
	}
}