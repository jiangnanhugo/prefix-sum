#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "info.h"


#define THREADS 256  /* Number of per-block threads */


int _length;         /* Length of the input array */
int _size;           /* Size of the input array in bytes */
int _blocks;         /* Number of GPU blocks to use */
int *h_input;        /* Host-side input array */
int *h_output;       /* Host-side output array */
int *d_input;        /* Device-side input array */
int *d_output;       /* Device-side output array */


/* Compute the prefix sum for each element in the block */
__global__ void compute_sums(int *input, int *output)
{
        int tid = threadIdx.x;
        int bid = blockIdx.x;
	int idx = (bid * THREADS) + tid;

        /* Initialize the buffers in shared memory */
        extern __shared__ int shmem[];
        int *in = shmem;
        int *out = &shmem[THREADS];
        in[tid] = input[idx];
        out[tid] = input[idx];
        __syncthreads();

        /* Compute the prefix sums */
        int offset;
        for (offset = 1; offset < THREADS; offset *= 2) {
                /* Swap the arrays */
                int *tmp = in;
                in = out;
                out = tmp;
                __syncthreads();

                if (tid - offset < 0)
                        out[tid] = in[tid];
                else    
                        out[tid] = in[tid] + in[tid - offset];
        }

	/* Copy the shared memory output to main memory */
	output[idx] = out[tid];
}

/* 
 * Add the highest prefix sum of block i-1 to each element
 * of block i
 */
__global__ void aggregate_blocks(int *output, int blocks)
{
	if (blocks == 1)
		return;

	int tid = threadIdx.x;
	int i;
	for (i = 1; i < blocks; i++) {
		int idx = (i * THREADS) + tid;
		int prev_block = (i * THREADS) - 1;
		output[idx] += output[prev_block];
		__syncthreads();
	}
}

/* Parse the input file */
__host__ void read_input(char *inputname)
{
        /* Open the input file */
        FILE *inputfile = fopen(inputname, "r");
        if (inputfile == NULL) {
                fprintf(stderr, "Invalid filename\n");
                free(inputname);
                exit(EXIT_FAILURE);
        }

        /* Read the line count */
        char *line = NULL;
        size_t len = 0;
        ssize_t read = getline(&line, &len, inputfile);
        _length = atoi(line);

	/* Compute the number of blocks to use */
	if (_length <= THREADS)
		_blocks = 1;
	else
		_blocks = _length / THREADS;

	/* Allocate the input/output arrays */
	_size = sizeof(int) * _length;
        h_input = (int *)malloc(_size);
	h_output = (int *)malloc(_size);

        /* Read the input */
        int i = 0;
        while ((read = getline(&line, &len, inputfile)) != -1) {
                int x = atoi(line);
                h_input[i] = x;
                i++;
        }

        /* Copy the input to the GPU */
        cudaMalloc((void **) &d_input, _size);
        cudaMemcpy(d_input, h_input, _size, 
                   cudaMemcpyHostToDevice);

	/* Allocate the output array on the GPU */
	cudaMalloc((void **) &d_output, _size);

        free(line);
        fclose(inputfile);
}

/* Print the prefix sums */
__host__ void print_results(int *output, int len)
{
        // int i;
        // for (i = 0; i < len; i++)
        //         printf("%d ", output[i]);
        // printf("\n");

	printf("Final prefix sum: %d\n", output[len - 1]);
}

__host__ int main(int argc, char *argv[])
{
        if (argc < 2) {
                fprintf(stderr, "Must provide a filename\n");
                return -1;
        }
        size_t len = strlen(argv[1]);
        char *inputname = (char *)malloc(len + 1);
        strcpy(inputname, argv[1]);
        read_input(inputname);
        
	/* Compute the prefix sums for each block */
        int shmem_size = sizeof(int) * THREADS * 2;
        compute_sums<<<_blocks, THREADS, shmem_size>>>(d_input, d_output);

	/* Compute the final results */
	aggregate_blocks<<<1, THREADS>>>(d_output, _blocks);
	cudaMemcpy(h_output, d_output, _size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	print_results(h_output, _length);

	free(inputname);
	free(h_input);
	free(h_output);
	cudaFree(d_input);
	cudaFree(d_output);
	return 0;
}
