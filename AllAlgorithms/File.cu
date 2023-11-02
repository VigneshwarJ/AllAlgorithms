#include "ParallelExamples.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__
void addVectors(float* A, float* B, float* C, size_t size)
{
	size_t i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		C[i] = A[i] + B[i];
	}

}

__global__
void multiplyMatrix(float* A, float* B, float* C, int m, int n, int o)
{
	size_t i = threadIdx.x + blockDim.x * blockIdx.x;
	size_t j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j < o)
	{
		float val = 0;
		for (size_t k = 0; k < n; k++)
		{
			val += A[i * m + k] *B[k*n+j];
		}
		C[i * m + j] = val;

	}
	/*size_t k = threadIdx.y + blockDim.y * blockIdx.y;*/
}


void addVectorSIMT(float* A, float* B, float* C, size_t size)
{
	//host program
	float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;

	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	addVectors << <(size + 2047) / 2048, 2048 >> > (d_A, d_B, d_C, size);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}
/* 2x3 A and 3x1 B
| a00*b00 + a01*b10 + a02*b20 |
| a10*b00 + a11*b10 + a12*b20 |
*/
void matrixMultiplySIMTNaive(float* A, float* B, float* C, int m, int n, int o)
{
	float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
	cudaMalloc((void**)&d_A, m*n * sizeof(float));
	cudaMalloc((void**)&d_B, n*o * sizeof(float));
	cudaMalloc((void**)&d_C, m*o * sizeof(float));
	dim3 threadsPerBlock(m, o);
	dim3 blocksPerGrid(1, 1);
	if (m*o > 512) {
		threadsPerBlock.x = 16;
		threadsPerBlock.y = 16;
		blocksPerGrid.x = ceil(double(m) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(o) / double(threadsPerBlock.y));
	}
	cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, n * o * sizeof(float), cudaMemcpyHostToDevice);
	multiplyMatrix << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, m, n, o);
	cudaMemcpy(C, d_C, m*o * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}