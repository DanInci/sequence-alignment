#include <stdio.h>
#include <cuda.h>

#include "kernel_functions.h"

#define CUDA_BLOCK_SIZE 1024

#define MATCH 1
#define MISMATCH -1
#define GAP -1

#define MAX(a,b) (((a)>(b))?(a):(b))
#define FIT_SCORE(a, b) (a == b ? MATCH : MISMATCH)

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
void cuda_init_score(int *score, int row_dim, int col_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < row_dim) {
        score[index] = index * GAP;
    }

    if (index < col_dim) {
        score[index * col_dim] = index * GAP;
    }
}

__global__
void cuda_compute_score(char *seqA, char *seqB, int *score, int row_dim, int col_dim) {
    int cuda_grid_size;

    // initialize first row & first column
    // all gaps => row index * GAP score
    // cuda_grid_size = MAX(row_dim, col_dim) / CUDA_BLOCK_SIZE + 1;
    // cuda_init_score<<<cuda_grid_size, CUDA_BLOCK_SIZE>>>(score, row_dim, col_dim);

    // wait for GPU
    // cudaDeviceSynchronize();

    // anti-diagonal traversal, except first row and column
    for (int slice = 2; slice < row_dim + col_dim - 1; slice++) {
        int z1 = slice <= col_dim ? 1 : slice - col_dim + 1;
        int z2 = slice <= row_dim ? 1 : slice - row_dim + 1;
        int slice_size = slice - z2 - z1 + 1;

        cuda_grid_size = slice_size / CUDA_BLOCK_SIZE + 1;
        cuda_compute_slice<<<cuda_grid_size, CUDA_BLOCK_SIZE>>>(seqA, seqB, score, col_dim, slice, z1, slice-z2);

        // wait for GPU to finish computing the anti-diagonal
        cudaDeviceSynchronize();
    }
}