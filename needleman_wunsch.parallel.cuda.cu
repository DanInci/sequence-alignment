#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
void cuda_compute_slice(char *seqA, char *seqB, int lenA, int lenB, int **score, int slice, int z1, int z2) {
    for (int j = slice - z2; j >= z1; j--) {
        char a = seqA[j-1];
        char b = seqB[slice-j-1];

        /*
        * MAX OF (FIT (MATCH OR MISMATCH), INSERT, DELETE) SCORES
        */
        int match_score = score[j-1][slice-j-1] + FIT_SCORE(a, b);
        int insert_score = score[j][slice-j-1] + GAP;
        int delete_score = score[j-1][slice-j] + GAP;

        int max = MAX(match_score, insert_score);
        max = MAX(max, delete_score);

        score[j][slice-j] = max;
    }
}

__global__
void cuda_test(int n, int *a, int *b) {
    for(int i=0; i<n; i++) {
        b[i] = a[i] + b[i];
    }
}

int **needleman_wunsch_score(char *seqA, char *seqB, int lenA, int lenB) {
    // char *cudaSeqA, *cudaSeqB;
    // int i, j, **score, **cuda_score;
    
    // allocate unified memory for sequences
    // cudaMallocManaged(&cudaSeqA, lenA * sizeof(char));
    // cudaMallocManaged(&cudaSeqB, lenB * sizeof(char));

    // cudaMemcpy(cudaSeqA, seqA, lenA * sizeof(char), cudaMemcpyHostToDevice);
    // cudaMemcpy(cudaSeqB, seqB, lenB * sizeof(char), cudaMemcpyHostToDevice);

    // // allocate unified memory for score matrix
    // score = (int **) calloc((lenA + 1), sizeof(int *));
    // cudaMallocManaged(&cuda_score, (lenA + 1) * sizeof(int *));
    // for (i = 0; i <= lenA; i++) {
    //     score[i] = (int *) calloc((lenB + 1), sizeof(int));
    //     cudaMallocManaged(&cuda_score[i], (lenB + 1) * sizeof(int));
    // }

    // // initlialize first column
    // // all gaps => column index * GAP score
    // for (i = 0; i <= lenA; i++) {
    //     score[i][0] = i * GAP;
    // }

    // // initialize first row
    // // all gaps => row index * GAP score
    // for (j = 0; j <= lenB; j++) {
    //     score[0][j] = j * GAP;
    // }

    // // copy score matrix into cuda memory
    // cudaMemcpy(cuda_score, score, (lenA + 1) * sizeof(int *), cudaMemcpyHostToDevice);
    // for (i = 0; i <= lenA; i++) {
    //     cudaMemcpy(cuda_score[i], score[i], (lenB + 1) * sizeof(int), cudaMemcpyHostToDevice);
    // }

    // int m = lenA + 1;
    // int n = lenB + 1;


    int **score, *x, *y, i;
    int n = 1 << 20;

    cudaMallocManaged(&x, n*sizeof(int));
    cudaMallocManaged(&y, n*sizeof(int));

    for(i=0;i<n;i++) {
        x[i] = 3;
        y[i] = 1;
    }
    cuda_test<<<256,256>>>(n, x, y);

    cudaDeviceSynchronize();

    // // anti-diagonal traversal, except first row and column
    // for (int slice = 2; slice < m + n - 1; slice++) {
    //     int z1 = slice <= n ? 1 : slice - n + 1;
    //     int z2 = slice <= m ? 1 : slice - m + 1;
    //     int size = slice - m + n + 1;

    //     // cuda_compute_slice<<<1, 1>>>(cudaSeqA, cudaSeqB, lenA, lenB, cuda_score, slice, z1, z2);

    //     // wait for GPU to finish computing the anti-diagonal
    //     cudaDeviceSynchronize();
    // }

    // // copy score matrix from cuda memory
    // cudaMemcpy(score, cuda_score, (lenA + 1) * sizeof(int *), cudaMemcpyDeviceToHost);
    // for (i = 0; i <= lenA; i++) {
    //     cudaMemcpy(score[i], cuda_score[i], (lenB + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    // }

    // // deallocate cuda shared sequences
    // for (i = 0; i <= lenA; i++) {
    //     cudaFree(score[i]);
    // }
    // cudaFree(score);
    // cudaFree(cudaSeqA);
    // cudaFree(cudaSeqB);

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
    int i, lenA, lenB;

    printf("Here");

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <in.seq>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    printf("Here");

    input = fopen(argv[1], "r");
    if (input == NULL) {
        fprintf(stderr, "Failed to open input file %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }

    printf("Here");

    read_sequence(input, &seqA_descriptor, &seqA, &lenA);
    read_sequence(input, &seqB_descriptor, &seqB, &lenB);

    printf("Here 1");

    alignment = needleman_wunsch_align(seqA, seqB, lenA, lenB);

    strcpy(output_filename, argv[1]);
    strcat(output_filename, ".aligned");
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

    //print score matrix
    // for (i = 0; i <= lenA; i++) {
    //     for (j = 0; j <= lenB; j++) {
    //         printf("%3d ", alignment->score[i][j]);
    //     }
    //     printf("\n");
    // }

    // print sequences alignment
    // printf("\nInitial:\n%s\n%s\n\n", seqA, seqB);
    // printf("Aliniere:\n%s\n%s\n\n", alignment->alignedA, alignment->alignedB);

    // deallocate memory
    for (i = 0; i <= lenA; i++) {
        cudaFree(alignment->score[i]);
    }
    cudaFree(alignment->score);
    free(alignment);
    free(seqA_descriptor);
    free(seqB_descriptor);
    free(seqA);
    free(seqB);


    fclose(input);
    fclose(output);

    exit(EXIT_SUCCESS);
}