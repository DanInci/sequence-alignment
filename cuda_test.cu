

__global__ void add(int n, int *x, int *y) {
    for(int i=0; i<n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N, *x, *y;

    N = 1 << 20;

    int error = cudaMallocManaged(&x, N*sizeof(int));
    fprintf("%d", error);
    cudaMallocManaged(&y, N*sizeof(int));

    for(int i=0; i<N; i++) {
        x[i] = 2;
        y[i] = 10;
    }

    add<<<1, 1>>>(N, x, y);

    cudaDeviceSynchronize();

    cudaFree(x);
    cudaFree(y);

    return 0;
}