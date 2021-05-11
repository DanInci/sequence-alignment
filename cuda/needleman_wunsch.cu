#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernel_functions.h"

#define MATCH 1
#define MISMATCH -1
#define GAP -2

#define FIT_SCORE(a, b) (a == b ? MATCH : MISMATCH)

struct Alignment {
    int **score;
    char *alignedA;
    char *alignedB;
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"\nGPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int **needleman_wunsch_score(char *seqA, char *seqB, int lenA, int lenB) {
    char *cudaSeqA, *cudaSeqB;
    int **score, *cuda_score;

    int row_dim = lenA + 1;
    int col_dim = lenB + 1;

    printf("Allocating CUDA memory ... ");
    
    // allocate unified memory for sequences & score matrix
    gpuErrchk(cudaMallocManaged(&cudaSeqA, lenA * sizeof(char)));
    gpuErrchk(cudaMallocManaged(&cudaSeqB, lenB * sizeof(char)));
    gpuErrchk(cudaMallocManaged(&cuda_score, row_dim * col_dim * sizeof(int)));

    gpuErrchk(cudaMemcpy(cudaSeqA, seqA, lenA * sizeof(char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudaSeqB, seqB, lenB * sizeof(char), cudaMemcpyHostToDevice));

    printf("DONE\n");
    printf("Calculating score matrix ... ");

    cuda_compute_score<<<1, 1>>>(cudaSeqA, cudaSeqB, cuda_score, row_dim, col_dim);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    printf("DONE\n");
    printf("CUDA cleanup ... ");

    // copy score matrix from cuda memory
    score = (int **) calloc(row_dim, sizeof(int *));
    for (int i = 0; i < row_dim; i++) {
        score[i] = (int *) calloc(col_dim, sizeof(int));
        gpuErrchk(cudaMemcpy(score[i], &cuda_score[i * row_dim], col_dim * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // deallocate cuda shared sequences & score matrix
    cudaFree(cudaSeqA);
    cudaFree(cudaSeqB);
    cudaFree(cuda_score);

    printf("DONE\n");

    return score;
}

struct Alignment *needleman_wunsch_align(char *seqA, char *seqB, int lenA, int lenB) {
    struct Alignment *alignment;
    char *alignedA, *alignedB;
    int i, j, k;

    // allocate memory
    alignment = (struct Alignment *) calloc(1, sizeof(struct Alignment *));
    k = lenA + lenB; // maximum length of alignment is lenA + lenB
    alignedA = (char *) calloc((k + 1), sizeof(char *));
    alignedA[k] = '\0';
    alignedB = (char *) calloc((k + 1), sizeof(char *));
    alignedB[k] = '\0';
    k--;

    alignment->score = needleman_wunsch_score(seqA, seqB, lenA, lenB);

    i=lenA;
    j=lenB;

    printf("Constructing alignment ... ");

    while (i > 0 && j > 0) {
        if (alignment->score[i][j] == alignment->score[i-1][j-1] + FIT_SCORE(seqA[i-1], seqB[j-1])) {
            alignedA[k] = seqA[i-1];
            alignedB[k] = seqB[j-1];
            i--;
            j--;
        }
        else if (alignment->score[i][j] == alignment->score[i-1][j] + GAP) {
            alignedA[k] = seqA[i-1];
            alignedB[k] = '-';
            i--;
        }
        else {
            alignedA[k] = '-';
            alignedB[k] = seqB[j-1];
            j--;

        }
        k--;
    }

    while (i > 0 || j > 0) {
        if (i == 0) {
            alignedA[k] = '-';
            alignedB[k] = seqB[j-1];
            j--;
        }
        else if (j == 0) {
            alignedA[k] = seqA[i-1];
            alignedB[k] = '-';
            i--;
        }
        k--;
    }

    alignment->alignedA = alignedA+k+1;
    alignment->alignedB = alignedB+k+1;

    printf("DONE\n");

    return alignment;
}


void read_sequence(FILE *in, char **descriptor, char **seq, int *len) {
    char *line = NULL;
    size_t size = 0;
    ssize_t read; 

    if ((read = getline(&line, &size, in)) != -1) {
        line[read-1] = '\0';

        *descriptor = (char *) calloc((size + 1), sizeof(char));
        strcpy(*descriptor, line);
        
        *len = 0;
        while ((read = getline(&line, &size, in)) != -1 && memchr(line, '>', sizeof(char)) == NULL) {
            line[read-2] = '\0';

            if (*len == 0) {
                *seq = (char *) calloc(size-2, sizeof(char));
                strcpy(*seq, line);
                *len = read-2;
            }
            else {
                *seq = (char *) realloc(*seq, (*len + read) * sizeof(char));
                strcat(*seq, line);
                *len += read-2;
            }
        }

        // unread last line if it is for another sequence
        if (memchr(line, '>', sizeof(char)) != NULL) {
            fseek(in, -1 * read, SEEK_CUR);
        }

    } else {
        fprintf(stderr, "Failed to read sequence\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    char output_filename[100];
    char *seqA_descriptor, *seqA, *seqB_descriptor, *seqB;
    struct Alignment *alignment;
    FILE *input, *output;
    int lenA, lenB;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <in.seq>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    input = fopen(argv[1], "r");
    if (input == NULL) {
        fprintf(stderr, "Failed to open input file %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }

    read_sequence(input, &seqA_descriptor, &seqA, &lenA);
    read_sequence(input, &seqB_descriptor, &seqB, &lenB);

    alignment = needleman_wunsch_align(seqA, seqB, lenA, lenB);

    strcpy(output_filename, argv[1]);
    strcat(output_filename, ".cuda.aligned");
    output = fopen(output_filename, "w");
    if (output == NULL) {
        fprintf(stderr, "Failed to open output file %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    fputs(seqA_descriptor, output);
    fputc('\n', output);
    fputs(alignment->alignedA, output);
    fputc('\n', output);

    fputs(seqB_descriptor, output);
    fputc('\n', output);
    fputs(alignment->alignedB, output);
    fputc('\n', output);

    // // print score matrix
    // for (int i = 0; i <= lenA; i++) {
    //     for (int j = 0; j <= lenB; j++) {
    //         printf("%3d ", alignment->score[i][j]);
    //     }
    //     printf("\n");
    // }

    // // print sequences alignment
    // printf("\nInitial:\n%s\n%s\n\n", seqA, seqB);
    // printf("Aliniere:\n%s\n%s\n\n", alignment->alignedA, alignment->alignedB);

    // deallocate memory
    for (int i = 0; i <= lenA; i++) {
        free(alignment->score[i]);
    }
    free(alignment->score);
    free(alignment);
    free(seqA_descriptor);
    free(seqB_descriptor);
    free(seqA);
    free(seqB);


    fclose(input);
    fclose(output);

    exit(EXIT_SUCCESS);
}