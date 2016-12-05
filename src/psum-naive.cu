#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "info.h"


#define THREADS 256


int length;              /* Length of the input array */
int bytes;               /* Size of the input array in bytes */
int blocks;              /* Number of GPU blocks to use */
int *h_input;            /* Host-side input array */
int *h_output;           /* Host-side output array */
int *d_input;            /* Device-side input array */
int *d_output;           /* Device-side output array */


/* Compute the prefix sum for each element in the block */
__global__ void compute_sums(int *input, int *output, int length, int offset)
{
        int x = threadIdx.x + (blockIdx.x * blockDim.x); 
        int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int idx = x + (y * blockDim.x * gridDim.y);
	if (idx >= length || idx - offset < 0)
		return;

	output[idx] = input[idx] + input[idx - offset];
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
        length = atoi(line);

	/* Compute the number of blocks to use */
	if (length <= THREADS)
		blocks = 1;
	else
		blocks = ceil(length / THREADS);

	/* Allocate the CPU buffers */
	bytes = sizeof(int) * length;
        h_input = (int *)malloc(bytes);
	h_output = (int *)malloc(bytes);

        /* Read the input */
        int i = 0;
        while ((read = getline(&line, &len, inputfile)) != -1) {
                int x = atoi(line);
                h_input[i] = x;
                i++;
        }

        /* Allocate the GPU buffers */
        cudaMalloc((void **) &d_input, bytes);
	cudaMalloc((void **) &d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_input, bytes, cudaMemcpyHostToDevice);

        free(line);
        fclose(inputfile);
}

__host__ void print_results(int *output, int length)
{
	int i;
	for (i = 0; i < length; i++)
		printf("%d ", output[i]);
	printf("\n ");
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
        
	/* Set the block and grid dimensions */
	dim3 grid(ceil(sqrt(blocks)), ceil(sqrt(blocks)));
	dim3 block(sqrt(THREADS), sqrt(THREADS)); 

	/* Compute the prefix sums, one level of the tree at a time */
	int offset;
	for (offset = 1; offset < length; offset *= 2) {
		/* Swap the array pointers for double buffering */
		int *tmp = d_input;
		d_input = d_output;
		d_output = tmp;
		
		compute_sums<<<grid, block>>>(d_input, d_output,
					      length, offset);
	}
	cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
	printf("Final prefix sum: %d\n", h_output[length - 1]);

	free(inputname);
	free(h_input);
	free(h_output);
	cudaFree(d_input);
	cudaFree(d_output);
	return 0;
}
