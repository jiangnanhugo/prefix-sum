#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int _size;     /* Size of the input */
int *h_input;  /* Host-side input array */
int *d_input;  /* Device-side input array */


/* Print the prefix sums */
__device__ void print_results(int *output, int size)
{
        int i;
        for (i = 0; i < size; i++)
                printf("%d ", output[i]);
        printf("\n");
}

/* Compute the prefix sum for each element in place  */
__global__ void compute_sums(int *input, int size)
{
	int tid = threadIdx.x;

	/* Initialize shared memory */
	extern __shared__ int shmem[];
	int *in = shmem;
	int *out = &shmem[size];
	in[tid] = input[tid];
	out[tid] = input[tid];
	__syncthreads();

	/* Compute the prefix sums */
        // int offset;
	// for (offset = 1; offset < size; offset *= 2) {
	// 	/* ... */

	// 	/* Swap the arrays */
	// 	int *tmp = in;
	// 	in = out;
	// 	out = tmp;
	// }
	// __syncthreads();
        
	if (tid == 0)
		print_results(out, size);
	__syncthreads();
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
        _size = atoi(line);
        h_input = (int *)malloc(sizeof(int) * _size);

        /* Read the integers */
        int i = 0;
        while ((read = getline(&line, &len, inputfile)) != -1) {
                int x = atoi(line);
                h_input[i] = x;
                i++;
        }

	/* Copy the input to the GPU */
	cudaMalloc((void **) &d_input, sizeof(int) * _size);
	cudaMemcpy(d_input, h_input, sizeof(int) * _size, 
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

	int shmem_size = sizeof(int) * _size * 2;
        compute_sums<<<1, _size, shmem_size>>>(d_input, _size);

        free(inputname);
        free(h_input);
	cudaFree(d_input);
        
	return 0;
}
