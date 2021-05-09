#ifndef kernel_functions
#define kernel_functions


/*
* MAX OF (FIT (MATCH OR MISMATCH), INSERT, DELETE) SCORES
*/
extern __global__
void cuda_compute_slice(char *seqA, char *seqB, int *score, int col_dim, int slice, int start, int end);

extern __global__
void cuda_init_score(int *score, int row_dim, int col_dim);

extern __global__
void cuda_compute_score(char *seqA, char *seqB, int *score, int row_dim, int col_dim);

#endif