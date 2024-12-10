#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NMAX 3200000

#define SAFE_CALL(CallInstruction) { \
    cudaError_t cuerr = CallInstruction; \
    if (cuerr != cudaSuccess) { \
        printf("CUDA error: %s at call \"%s\"\n", cudaGetErrorString(cuerr), #CallInstruction); \
        exit(0); \
    } \
}

__global__ void addKernel(double* sum_GPU, double* mass_1_device, double* mass_2_device, double* mass_3_device, double* mass_4_device, double* mass_5_device, double* mass_6_device, double* mass_7_device, int size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
		sum_GPU[i] = mass_1_device[i] + mass_2_device[i] + mass_3_device[i] + mass_4_device[i] + mass_5_device[i] + mass_6_device[i] + mass_7_device[i];
}

int main(int argc, char* argv[])
{
	int GridDim = 1024, BlockDim = 128;
	cudaEvent_t start_event, stop_event;

	double start_time = 0, t_s = 0, t_tr = 0, t_cu = 0; 
	float GPU_time = 0;

	int rep = 0; const int reps = 20;

	double* mass_1 = (double*)malloc(NMAX * sizeof(double));
	double* mass_2 = (double*)malloc(NMAX * sizeof(double));
	double* mass_3 = (double*)malloc(NMAX * sizeof(double));
	double* mass_4 = (double*)malloc(NMAX * sizeof(double));
	double* mass_5 = (double*)malloc(NMAX * sizeof(double));
	double* mass_6 = (double*)malloc(NMAX * sizeof(double));
	double* mass_7 = (double*)malloc(NMAX * sizeof(double));

	double* sum_sequence = (double*)malloc(NMAX * sizeof(double));
	double* sum_host = (double*)malloc(NMAX * sizeof(double));

	for (int i = 0; i < NMAX; ++i) 
	{
		mass_1[i] = 1;
		mass_2[i] = 1;
		mass_3[i] = 1;
		mass_4[i] = 1;
		mass_5[i] = 1;
		mass_6[i] = 1;
		mass_7[i] = 1;
	}

	double* mass_1_device; double* mass_2_device; double* mass_3_device; double* mass_4_device;
	double* mass_5_device; double* mass_6_device; double* mass_7_device; 
	double* sum_GPU;

	
	SAFE_CALL(cudaMalloc((void**)&mass_1_device, NMAX * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&mass_2_device, NMAX * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&mass_3_device, NMAX * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&mass_4_device, NMAX * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&mass_5_device, NMAX * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&mass_6_device, NMAX * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&mass_7_device, NMAX * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&sum_GPU, NMAX * sizeof(double)));


	SAFE_CALL(cudaEventCreate(&start_event));
	SAFE_CALL(cudaEventCreate(&stop_event));

	for (rep = 0; rep < reps; rep++)
	{
		start_time = clock();
		for (int i = 0; i < NMAX; ++i)
		{
			sum_sequence[i] = mass_1[i] + mass_2[i] + mass_3[i] + mass_4[i] + mass_5[i] + mass_6[i] + mass_7[i];
		}
		t_s += (clock() - start_time) / CLOCKS_PER_SEC;

		start_time = clock();
		SAFE_CALL(cudaMemcpy(mass_1_device, mass_1, NMAX * sizeof(double), cudaMemcpyHostToDevice));
		SAFE_CALL(cudaMemcpy(mass_2_device, mass_2, NMAX * sizeof(double), cudaMemcpyHostToDevice));
		SAFE_CALL(cudaMemcpy(mass_3_device, mass_3, NMAX * sizeof(double), cudaMemcpyHostToDevice));
		SAFE_CALL(cudaMemcpy(mass_4_device, mass_4, NMAX * sizeof(double), cudaMemcpyHostToDevice));
		SAFE_CALL(cudaMemcpy(mass_5_device, mass_5, NMAX * sizeof(double), cudaMemcpyHostToDevice));
		SAFE_CALL(cudaMemcpy(mass_6_device, mass_6, NMAX * sizeof(double), cudaMemcpyHostToDevice));
		SAFE_CALL(cudaMemcpy(mass_7_device, mass_7, NMAX * sizeof(double), cudaMemcpyHostToDevice));
		t_tr += (clock() - start_time);

		SAFE_CALL(cudaEventRecord(start_event, 0));

		addKernel << <GridDim, BlockDim >> > (sum_GPU, mass_1_device, mass_2_device, mass_3_device, mass_4_device, mass_5_device, mass_6_device, mass_7_device, NMAX);

		SAFE_CALL(cudaEventRecord(stop_event, 0));
		SAFE_CALL(cudaEventSynchronize(stop_event));

		SAFE_CALL(cudaEventElapsedTime(&GPU_time, start_event, stop_event));
		t_cu += GPU_time / 1000;

		start_time = clock();
		SAFE_CALL(cudaMemcpy(sum_host, sum_GPU, NMAX * sizeof(double), cudaMemcpyDeviceToHost));
		t_tr += (clock() - start_time);
	}

	t_tr /= CLOCKS_PER_SEC;
	t_s /= reps; t_tr /= reps; t_cu /= reps;

	printf("Sum sequence (0, 1, ... , NMAX-2, NMAX-1): %.1f %.1f ... %.1f %.1f\n", sum_sequence[0], sum_sequence[1], sum_sequence[NMAX - 2], sum_sequence[NMAX - 1]);
	printf("Sum CUDA (0, 1, ... , NMAX-2, NMAX-1): %.1f %.1f ... %.1f %.1f\n", sum_host[0], sum_host[1], sum_host[NMAX - 2], sum_host[NMAX - 1]);

	printf("\nAverage time:\n");
	printf("Sequence algorithm time: %f seconds.\n", t_s);
	printf("Transfer time: %f seconds\n", t_tr);
	printf("Kernel execution time: %f seconds\n", t_cu);

	double a_cu = t_s / t_cu; 
	double a_cutr = t_s / (t_cu + t_tr);
	
	printf("\nCalculating acceleration:\n");
	printf("Kernel execution: %f \n", a_cu);
	printf("Kernel execution + transfer time: %f \n", a_cutr);

	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

	cudaFree(mass_1_device);
	cudaFree(mass_2_device);
	cudaFree(mass_3_device);
	cudaFree(mass_4_device);
	cudaFree(mass_5_device);
	cudaFree(mass_6_device);
	cudaFree(mass_7_device);
	cudaFree(sum_GPU);

	free(sum_sequence);
	free(mass_1);
	free(mass_2);
	free(mass_3);
	free(mass_4);
	free(mass_5);
	free(mass_6);
	free(mass_7);
	free(sum_host);
}