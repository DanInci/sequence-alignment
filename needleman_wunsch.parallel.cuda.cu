#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_BLOCK_SIZE 256

#define MATCH 1
#define MISMATCH -1
#define GAP -1

#define MAX(a,b) (((a)>(b))?(a):(b))
#define FIT_SCORE(a, b) (a == b ? MATCH : MISMATCH)

struct Alignment {
    int **score;
    char *alignedA;
    char *alignedB;
};

__global__
void cuda_compute_slice(char *seqA, char *seqB, int *score, int col_dim, int slice, int start, int end) {
    int index = start + blockIdx.x * blockDim.x + threadIdx.x;

    if (index <= end) {
        char a = seqA[index-1];
        char b = seqB[slice-index-1];

        /*
        * MAX OF (FIT (MATCH OR MISMATCH), INSERT, DELETE) SCORES
        */
        int match_score = score[(col_dim * (index-1)) + slice-index-1] + FIT_SCORE(a, b);
        int insert_score = score[(col_dim * index) + slice-index-1] + GAP;
        int delete_score = score[(col_dim * (index-1)) + slice-index] + GAP;

        int max = MAX(match_score, insert_score);
        max = MAX(max, delete_score);

        score[(col_dim * index) + slice - index] = max;
    }
}

__global__
void cuda_init_row(int *score, int end) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index <= end) {
        score[index] = index * GAP;
    }
}

__global__
void cuda_init_column(int *score, int col_dim, int end) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index <= end) {
        score[index * col_dim] = index * GAP;
    }
}


int **needleman_wunsch_score(char *seqA, char *seqB, int lenA, int lenB) {
    char *cudaSeqA, *cudaSeqB;
    int **score, *cuda_score, cuda_grid_size;
    
    // allocate unified memory for sequences & score matrix
    cudaMallocManaged(&cudaSeqA, lenA * sizeof(char));
    cudaMallocManaged(&cudaSeqB, lenB * sizeof(char));
    cudaMallocManaged(&cuda_score, (lenA + 1) * (lenB + 1) * sizeof(int));

    cudaMemcpy(cudaSeqA, seqA, lenA * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaSeqB, seqB, lenB * sizeof(char), cudaMemcpyHostToDevice);

    int m = lenA + 1;
    int n = lenB + 1;

    printf("Done 0\n");

    // initialize first row
    // all gaps => row index * GAP score
    cuda_grid_size = m / CUDA_BLOCK_SIZE + 1;
    cuda_init_row<<<cuda_grid_size, CUDA_BLOCK_SIZE>>>(cuda_score, lenB);

    // initlialize first column
    // all gaps => column index * GAP score
    cuda_grid_size = n / CUDA_BLOCK_SIZE + 1;
    cuda_init_column<<<cuda_grid_size, CUDA_BLOCK_SIZE>>>(cuda_score, n, lenA);

    // wait for GPU
    cudaDeviceSynchronize();

    printf("Done 1\n");


    // anti-diagonal traversal, except first row and column
    for (int slice = 2; slice < m + n - 1; slice++) {
        int z1 = slice <= n ? 1 : slice - n + 1;
        int z2 = slice <= m ? 1 : slice - m + 1;
        int slice_size = slice - z2 - z1 + 1;

        cuda_grid_size = slice_size / CUDA_BLOCK_SIZE + 1;
        cuda_compute_slice<<<cuda_grid_size, CUDA_BLOCK_SIZE>>>(cudaSeqA, cudaSeqB, cuda_score, n, slice, z1, slice-z2);

        // wait for GPU to finish computing the anti-diagonal
        cudaDeviceSynchronize();
    }

    printf("Done 2\n");

    // copy score matrix from cuda memory
    score = (int **) calloc((lenA + 1), sizeof(int *));
    for (int i = 0; i <= lenA; i++) {
        score[i] = (int *) calloc((lenB + 1), sizeof(int));
        cudaMemcpy(score[i], &cuda_score[i * (lenB + 1)], (lenB + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // deallocate cuda shared sequences & score matrix
    cudaFree(cudaSeqA);
    cudaFree(cudaSeqB);
    cudaFree(cuda_score);

    printf("Done 3\n");

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