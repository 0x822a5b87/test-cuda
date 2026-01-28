#include <sgemm_util.cuh>

void run(const int M, const int N, const int K)
{
    host_sgemm_t *h_sgemm = allocate_host(M, N, K);
    device_sgemm_t *d_sgemm = allocate_device(M, N, K);

    // Initialize matrices on host
    init_matrix(h_sgemm);

    // Copy matrices from host to device
    memcpy_host_to_device(d_sgemm, h_sgemm);

    // Launch SGEMM kernel
    launch_sgemm_kernel(d_sgemm);

    // Copy result from device to host
    memcpy_device_to_host(h_sgemm, d_sgemm);

    verify_result(h_sgemm);

    // Free memory
    free_device(d_sgemm);
    free_host(h_sgemm);
}

host_sgemm_t *allocate_host(const int M, const int N, const int K)
{
    host_sgemm_t *h_sgemm = (host_sgemm_t *)malloc(sizeof(host_sgemm_t));

    float *h_A = allocateMatricOnHost(M, K);
    float *h_B = allocateMatricOnHost(K, N);
    float *h_C = allocateMatricOnHost(M, N);

    h_sgemm->A = h_A;
    h_sgemm->B = h_B;
    h_sgemm->C = h_C;
    h_sgemm->M = M;
    h_sgemm->N = N;
    h_sgemm->K = K;

    return h_sgemm;
}

void free_host(host_sgemm_t *h_sgemm)
{
    free(h_sgemm->A);
    free(h_sgemm->B);
    free(h_sgemm->C);
    free(h_sgemm);
}

device_sgemm_t *allocate_device(const int M, const int N, const int K)
{
    device_sgemm_t *d_sgemm_handler = (device_sgemm_t *)malloc(sizeof(device_sgemm_t));
    float *d_A;
    float *d_B;
    float *d_C;
    CHECK(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

    d_sgemm_handler->A = d_A;
    d_sgemm_handler->B = d_B;
    d_sgemm_handler->C = d_C;
    d_sgemm_handler->M = M;
    d_sgemm_handler->N = N;
    d_sgemm_handler->K = K;

    return d_sgemm_handler;
}

void free_device(device_sgemm_t *d_sgemm_handler)
{
    CHECK(cudaFree(d_sgemm_handler->A));
    CHECK(cudaFree(d_sgemm_handler->B));
    CHECK(cudaFree(d_sgemm_handler->C));
    free(d_sgemm_handler);
}

void memcpy_host_to_device(device_sgemm_t *d_sgemm, host_sgemm_t *h_sgemm)
{
    CHECK(cudaMemcpy(d_sgemm->A, h_sgemm->A, h_sgemm->M * h_sgemm->K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sgemm->B, h_sgemm->B, h_sgemm->K * h_sgemm->N * sizeof(float), cudaMemcpyHostToDevice));
}

void memcpy_device_to_host(host_sgemm_t *h_sgemm, device_sgemm_t *d_sgemm)
{
    CHECK(cudaMemcpy(h_sgemm->C, d_sgemm->C, h_sgemm->M * h_sgemm->N * sizeof(float), cudaMemcpyDeviceToHost));
}

// init_matrix init matrix A and B with pre-specified values:
// A[i][j] = i + j
// B[i][j] = i - j
// in this case, we can easily verify the correctness of matrix C
// C[i][j] = sum_{k=0}^{K-1} A[i][k] * B[k][j] = sum_{k=0}^{K-1} (i + k) * (k - j)
void init_matrix(host_sgemm_t *h_sgemm)
{
    for (int i = 0; i < h_sgemm->M; i++)
    {
        for (int j = 0; j < h_sgemm->K; j++)
        {
            h_sgemm->A[i * h_sgemm->K + j] = static_cast<float>(i + j);
        }
    }

    for (int i = 0; i < h_sgemm->K; i++)
    {
        for (int j = 0; j < h_sgemm->N; j++)
        {
            h_sgemm->B[i * h_sgemm->N + j] = static_cast<float>(i - j);
        }
    }
}

float get_theoretical_result(int i, int j, int K)
{
    double k_double = (double)K;
    double term1 = (k_double * (k_double - 1.0f) * (2.0f * k_double - 1.0f)) / 6.0f;
    double term2 = (double)(i - j) * (k_double * (k_double - 1.0f)) / 2.0f;
    double term3 = k_double * i * j;

    return (float)(term1 + term2 - term3);
}

void verify_result(host_sgemm_t *h_sgemm)
{
    bool correct = true;
    for (int i = 0; i < h_sgemm->M; i++)
    {
        for (int j = 0; j < h_sgemm->N; j++)
        {
            float expected = get_theoretical_result(i, j, h_sgemm->K);
            float actual = h_sgemm->C[i * h_sgemm->N + j];
            if (!is_nearly_equal(expected, actual, h_sgemm->K))
            {
                printf("Mismatch at C[%d][%d]: expected %f, got %f\n", i, j, expected, actual);
                correct = false;
            }
        }
        if (!correct)
        {
            break;
        }
    }
    if (correct)
    {
        printf("Result verification passed!\n");
    }
}