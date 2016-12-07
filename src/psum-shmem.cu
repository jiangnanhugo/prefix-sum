#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "info.h"


#define THREADS 256


int length;              /* Length of the input array */
int blocks;              /* Number of GPU blocks to use */
int buffers_size;        /* Size of the input/output buffers in bytes */
int scalars_size;        /* Size of the scalars array in bytes */
int shmem_size;          /* Size of shared memory in bytes */
int *h_input;            /* Host-side input array */
int *h_scalars;          /* Host-side scalars for each block */
int *h_output;           /* Host-side output array */
int *d_input;            /* Device-side input array */
int *d_scalars;          /* Device-side scalars for each block */
int *d_output;           /* Device-side output array */


/* Compute partial prefix sums on the GPU */
__global__ void compute_sums(int *input, int *output, int *scalars, int length)
{
        int tid = threadIdx.x;
        int bid = blockIdx.x + (blockIdx.y * gridDim.x);
        int idx = tid + (bid * blockDim.x);
        if (idx >= length)
                return;

        /* Load input into shared memory */
        extern __shared__ int temp[];
        int *in = temp;
        int *out = &temp[THREADS];
        in[tid] = input[idx];
        out[tid] = input[idx];
        __syncthreads();
        
        /* Compute the partial sums */
        int offset;
        for (offset = 1; offset < THREADS; offset *= 2) {
                /* Swap the buffer pointers */
                int *swap = in;
                in = out;
                out = swap;
                
                if (tid < offset)
                        out[tid] = in[tid];
                else
                        out[tid] = in[tid] + in[tid - offset];
                __syncthreads();
        }
        output[idx] = out[tid];

        /* Copy the highest sum to the array of block sums */
        if (tid == 0)
                scalars[bid] = out[THREADS - 1];
}

/* Add the scalar to each element in the block */
__global__ void add_scalars(int *output, int *scalars, int length)
{
        int tid = threadIdx.x;
        int bid = blockIdx.x + (blockIdx.y * gridDim.x);
        int idx = tid + (bid * blockDim.x);
        if (idx >= length)
                return;

        extern __shared__ int scalar[];
        scalar[0] = scalars[bid - 1];

        if (bid > 0)
                output[idx] += scalar[0];
}

/* Aggregate the scalars on the CPU */
__host__ void compute_scalars(int *scalars, int length)
{
        int i;
        for (i = 1; i < length; i++)
                scalars[i] += scalars[i - 1]; 
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
        scalars_size = sizeof(int) * blocks;
        shmem_size = sizeof(int) * THREADS * 2;

        /* Allocate the CPU buffers */
        buffers_size = sizeof(int) * length;
        h_input = (int *)malloc(buffers_size);
        h_output = (int *)malloc(buffers_size);
        h_scalars = (int *)malloc(scalars_size);
        
        /* Read the input */
        int i = 0;
        while ((read = getline(&line, &len, inputfile)) != -1) {
                int x = atoi(line);
                h_input[i] = x;
                i++;
        }

        /* Allocate the GPU buffers */
        cudaMalloc((void **) &d_input, buffers_size);
        cudaMalloc((void **) &d_output, buffers_size);
        cudaMalloc((void **) &d_scalars, scalars_size);
        cudaMemcpy(d_input, h_input, buffers_size, cudaMemcpyHostToDevice);
        
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
        
        /* Compute the partial sums */
        dim3 grid(ceil(sqrt(blocks)), ceil(sqrt(blocks)));
        dim3 block(THREADS, 1);
        compute_sums<<<grid, block, shmem_size>>>(d_input, d_output,
                                                  d_scalars, length);
        cudaMemcpy(h_scalars, d_scalars, scalars_size,
                   cudaMemcpyDeviceToHost);

        /* Compute the scalars for each block */
        compute_scalars(h_scalars, blocks);
        cudaMemcpy(d_scalars, h_scalars, scalars_size,
                   cudaMemcpyHostToDevice);

        /* Add the scalars for each block to the output */
        add_scalars<<<grid, block, sizeof(int)>>>(d_output, d_scalars, length);
        cudaMemcpy(h_output, d_output, buffers_size, cudaMemcpyDeviceToHost);
        printf("Final prefix sum: %d\n", h_output[length - 1]);
        
        free(inputname);
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
        return 0;
}
