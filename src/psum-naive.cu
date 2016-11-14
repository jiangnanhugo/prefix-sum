#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int _length;       /* Length of the input array */
int _blocks;       /* Number of GPU blocks to use */
int _input_size;   /* Size of the input array in bytes */
int _sums_size;    /* Size of the block sums array in bytes */
int *h_input;      /* Host-side input array */
int *h_block_sums; /* Host-side sums for each block */ 
int *d_input;      /* Device-side input array */
int *d_block_sums; /* Device-side sums for each block */


/* Print the prefix sums */
__device__ void print_results(int *output, int size)
{
        int i;
        for (i = 0; i < size; i++)
                printf("%d ", output[i]);
        printf("\n");
}

/* Compute the prefix sum for each element in the block */
__global__ void compute_sums(int *input, int *block_sums, int size)
{
        int tid = threadIdx.x;
        int bid = blockIdx.x;

        /* Initialize the buffers in shared memory */
        extern __shared__ int shmem[];
        int *in = shmem;
        int *out = &shmem[size];
        in[tid] = input[tid];
        out[tid] = input[tid];
        __syncthreads();

        /* Compute the prefix sums */
        int offset;
        for (offset = 1; offset < size; offset *= 2) {
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
	__syncthreads();

	/* Copy the highest prefix sum to the block sums array */
	if (tid == 0)
		block_sums[bid] = out[size - 1];
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
	_blocks = 256 / _length; /* TODO: Use max threads per block */
	_input_size = sizeof(int) * _length;
	_sums_size = sizeof(int) * _blocks;
        h_input = (int *)malloc(_input_size);
	h_block_sums = (int *)malloc(_sums_size);

        /* Read the integers */
        int i = 0;
        while ((read = getline(&line, &len, inputfile)) != -1) {
                int x = atoi(line);
                h_input[i] = x;
                i++;
        }

        /* Copy the input to the GPU */
        cudaMalloc((void **) &d_input, _input_size);
	cudaMalloc((void **) &d_block_sums, _sums_size);
        cudaMemcpy(d_input, h_input, _input_size, 
                   cudaMemcpyHostToDevice);
	cudaMemcpy(d_block_sums, h_block_sums, _sums_size,
		   cudaMemcpyHostToDevice);

        free(line);
        fclose(inputfile);
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
        int shmem_size = sizeof(int) * _length * 2;
        compute_sums<<<1, _length, shmem_size>>>(d_input, d_block_sums,
						 _length);
	cudaMemcpy(h_block_sums, d_block_sums, _sums_size,
		   cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	/* Get the final results */
	printf("h_block_sums[0] = %d\n", h_block_sums[0]);
        /* ... */

        free(inputname);
        free(h_input);
	free(h_block_sums);
        cudaFree(d_input);
	cudaFree(d_block_sums);
        return 0;
}
