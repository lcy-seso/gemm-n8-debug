#pragma once

#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() { return error_buf; }

extern "C" int init() {
    error_buf[0] = '\0';

    return 0;
}

float rand_normal(float mean = 0.0f, float stddev = 1.0f) {
    // Box-Muller transform to generate random numbers with Normal distribution
    float u1 = ((float)rand()) / (float)RAND_MAX;
    float u2 = ((float)rand()) / (float)RAND_MAX;

    // Avoid log(0) by ensuring u1 is not zero
    if (u1 < 1e-10f) u1 = 1e-10f;

    // Compute Gaussian random value
    float r = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);

    // Scale and shift to get desired mean and standard deviation
    return mean + stddev * r;
}

void print_matrix(const __half* data, int numel) {
    for (int i = 0; i < numel; ++i) {
        printf("%.0f, ", __half2float(data[i]));

        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
}
