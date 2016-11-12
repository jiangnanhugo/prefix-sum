#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int _size;    /* Size of the input */
int *_input;  /* Input array of integers */


/* Parse the input file */
void read_input(char *inputname)
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
        _input = malloc(sizeof(int) * _size);

        /* Read the integers */
        int i = 0;
        while ((read = getline(&line, &len, inputfile)) != -1) {
                int x = atoi(line);
                _input[i] = x;
                i++;
        }

        free(line);
        fclose(inputfile);
}

/* Compute the prefix sum for each element in place  */
void compute_sums()
{
        int i;
        for (i = 1; i < _size; i++)
                _input[i] = _input[i] + _input[i - 1];
}

/* Print the prefix sums */
void print_results()
{
        int i;
        for (i = 0; i < _size; i++)
                printf("%d ", _input[i]);
        printf("\n");
}

void main(int argc, char *argv[])
{
        if (argc < 2) {
                fprintf(stderr, "Must provide a filename\n");
                exit(EXIT_FAILURE);
        }
        size_t len = strlen(argv[1]);
        char *inputname = malloc(len + 1);
        strcpy(inputname, argv[1]);
       
        read_input(inputname);
        compute_sums();
        print_results();

        free(inputname);
        free(_input);
        exit(EXIT_SUCCESS);
}
