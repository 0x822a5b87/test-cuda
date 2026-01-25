#include <stdio.h>
#include <stdlib.h>
#include <chrono>

void add(float *x, float *y, float *r, int n) {
    for (int i = 0; i < n; i++) {
        *(r + i) = *(x + i) + *(y + i);
    }
}

void call_add() {
    int N = 1000000;
    size_t mem_size = sizeof(float) * N;
    float* x, *y, *r;

    x = static_cast<float*>(malloc(mem_size));
    y = static_cast<float*>(malloc(mem_size));
    r = static_cast<float*>(malloc(mem_size));

    for (int i = 0; i < N; i++) {
        *(x + i) = 1.0;
        *(y + i) = 2.0;
    }

    add(x, y, r, N);

    for (int i = 0; i < 10; i++) {
        printf("r[%d] = %.3f\n", i, *(r + i));
    }

    free(x);
    free(y);
    free(r);
}

int main(int argc, char const *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    call_add();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("Execution time: %f ms\n", elapsed.count());
    return 0;
}
