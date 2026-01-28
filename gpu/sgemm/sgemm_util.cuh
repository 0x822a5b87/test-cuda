#include <util.cuh>

struct host_sgemm_t
{
    float *A;
    float *B;
    float *C;
    int M;
    int N;
    int K;
};

struct device_sgemm_t
{
    float *A;
    float *B;
    float *C;
    int M;
    int N;
    int K;
};

inline bool is_nearly_equal(float a, float b, int K)
{
    float diff = fabsf(a - b);
    float abs_a = fabsf(a);
    float abs_b = fabsf(b);
    float max_val = (abs_a > abs_b) ? abs_a : abs_b;

    float abs_tol = K * 1e-6f; 
    if (diff < abs_tol) return true;

    return (diff / max_val) < 1e-3f;
}

host_sgemm_t *allocate_host(const int M, const int N, const int K);
void free_host(host_sgemm_t *h_sgemm);
device_sgemm_t *allocate_device(const int M, const int N, const int K);
void free_device(device_sgemm_t *d_sgemm_handler);
void memcpy_host_to_device(device_sgemm_t *d_sgemm, host_sgemm_t *h_sgemm);
void memcpy_device_to_host(host_sgemm_t *h_sgemm, device_sgemm_t *d_sgemm);
void init_matrix(host_sgemm_t *h_sgemm);
float get_theoretical_result(int i, int j, int K);
void verify_result(host_sgemm_t *h_sgemm);
void launch_sgemm_kernel(const device_sgemm_t *d_sgemm);
void run(const int M, const int N, const int K);
