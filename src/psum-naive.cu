#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int _length;       /* Length of the input array */
int _size;         /* Size of the input array in bytes */
int _blocks;       /* Number of GPU blocks to use */
int *h_input;      /* Host-side input array */
int *h_output;     /* Host-side output array */
int *d_input;      /* Device-side input array */
int *d_output;     /* Device-side output array */


/* Print the prefix sums */
__device__ void print_results(int *output, int len)
{
        int i;
        for (i = 0; i < len; i++)
                printf("%d ", output[i]);
        printf("\n");
}

/* Compute the prefix sum for each element in the block */
__global__ void compute_sums(int *input, int *output, int len)
{
        int tid = threadIdx.x;
        int bid = blockIdx.x;
	int idx = (bid * len) + tid;

        /* Initialize the buffers in shared memory */
        extern __shared__ int shmem[];
        int *in = shmem;
        int *out = &shmem[len];
        in[idx] = input[idx];
        out[idx] = input[idx];
        __syncthreads();

        /* Compute the prefix sums */
        int offset;
        for (offset = 1; offset < len; offset *= 2) {
                /* Swap the arrays */
                int *tmp = in;
                in = out;
                out = tmp;
                __syncthreads();

                if (tid - offset < 0)
                        out[idx] = in[idx];
                else    
                        out[idx] = in[idx] + in[idx - offset];
        }
	output[idx] = out[idx];

	if (idx == 0)
		print_results(output, len * 2);
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
	_blocks = _length / 4; /* TODO: Use max threads per block */
	_size = sizeof(int) * _length;
        h_input = (int *)malloc(_size);
	h_output = (int *)malloc(_size);

        /* Read the integers */
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
        int shmem_size = _size * 2;
        compute_sums<<<_blocks, _length, shmem_size>>>(d_input, d_output,
						       _length / _blocks);

	/* Compute the final results */
	/* ... */

	free(inputname);
	free(h_input);
	free(h_output);
	cudaFree(d_input);
	cudaFree(d_output);
	return 0;
}
