#include <iostream>
#include "catch.hpp"
#include "cuda_runtime.h"
#include "cublas_v2.h"

TEST_CASE("cublasSgemv test 1.", "[cublas]") {
    INFO("in_channels < out_channels");

    cublasHandle_t handle;
    REQUIRE( cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);

    const size_t batch_size = 1, n_in_channels = 3, width = 4;
    const size_t n_out_channels = 5;
    const size_t n_rows = n_in_channels, n_cols = n_out_channels;

    float *x = nullptr;
    REQUIRE( cudaMallocManaged(&x, batch_size * n_in_channels * width * sizeof(float)) == cudaSuccess );
    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_in_channels; ++ch) {
            for (int w = 0; w < width; ++w) {
                x[w + ch * width + batch * n_in_channels * width] = 1.0 * w + ch * 10.0;
            }
        }
    }

    float *A = nullptr;
    REQUIRE( cudaMallocManaged(&A, n_rows * n_cols * sizeof(float)) == cudaSuccess );
    for(int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < n_cols; ++col) {
            A[col + row * n_cols] = 1.0;
        }
    }

    float *y = nullptr;
    REQUIRE( cudaMallocManaged(&y, batch_size * n_out_channels * width * sizeof(float)) == cudaSuccess );
    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_out_channels; ++ch) {
            for (int w = 0; w < width; ++w) {
                y[w + ch * width + batch * n_out_channels * width] = 0.0;
            }
        }
    }

    const float alpha = 1.0, beta = 0.0;
    for(int batch = 0; batch < batch_size; batch++) {
        for(int w = 0; w < width; w ++) {
            REQUIRE( cublasSgemv_v2(handle, CUBLAS_OP_T,
                                    n_rows, n_cols, &alpha, A, n_rows,
                                    x + w, width, &beta, y + w, width) == CUBLAS_STATUS_SUCCESS );
        }
    }
    cudaDeviceSynchronize();

    for(int ch = 0; ch < n_out_channels; ++ch) {
        for (int w = 0; w < width; ++w) {
            std::cout << y[w + width * ch] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    float pat[width] = { 30.0, 33.0, 36.0, 39.0 };
    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_out_channels; ++ch) {
            for (int w = 0; w < width; ++w) {
                CHECK(y[w + ch * width + batch * n_out_channels * width] == pat[w]);
            }
        }
    }

    REQUIRE( cudaFree(A) == cudaSuccess);
    REQUIRE( cudaFree(x) == cudaSuccess);
    REQUIRE( cudaFree(y) == cudaSuccess);
    REQUIRE( cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("cublasSgemv test 2.", "[cublas]") {
    INFO("in_channels > out_channels");

    cublasHandle_t handle;
    REQUIRE( cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);

    const size_t batch_size = 1, n_in_channels = 5, width = 4;
    const size_t n_out_channels = 3;
    const size_t n_rows = n_in_channels, n_cols = n_out_channels;

    float *x = nullptr;
    REQUIRE( cudaMallocManaged(&x, batch_size * n_in_channels * width * sizeof(float)) == cudaSuccess );
    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_in_channels; ++ch) {
            for (int w = 0; w < width; ++w) {
                x[w + ch * width + batch * n_in_channels * width] = 1.0 * w + ch * 10.0;
            }
        }
    }

    float *A = nullptr;
    REQUIRE( cudaMallocManaged(&A, n_rows * n_cols * sizeof(float)) == cudaSuccess );
    for(int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < n_cols; ++col) {
            A[col + row * n_cols] = 1.0;
        }
    }

    float *y = nullptr;
    REQUIRE( cudaMallocManaged(&y, batch_size * n_out_channels * width * sizeof(float)) == cudaSuccess );
    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_out_channels; ++ch) {
            for (int w = 0; w < width; ++w) {
                y[w + ch * width + batch * n_out_channels * width] = 0.0;
            }
        }
    }

    const float alpha = 1.0, beta = 0.0;
    for(int batch = 0; batch < batch_size; batch++) {
        for(int w = 0; w < width; w ++) {
            REQUIRE( cublasSgemv_v2(handle, CUBLAS_OP_T,
                                    n_rows, n_cols, &alpha, A, n_rows,
                                    x + w, width, &beta, y + w, width) == CUBLAS_STATUS_SUCCESS );
        }
    }
    cudaDeviceSynchronize();

    for(int ch = 0; ch < n_out_channels; ++ch) {
        for (int w = 0; w < width; ++w) {
            std::cout << y[w + width * ch] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    float pat[width] = { 100.0, 105.0, 110.0, 115.0 };
    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_out_channels; ++ch) {
            for (int w = 0; w < width; ++w) {
                CHECK(y[w + ch * width + batch * n_out_channels * width] == pat[w]);
            }
        }
    }

    REQUIRE( cudaFree(A) == cudaSuccess);
    REQUIRE( cudaFree(x) == cudaSuccess);
    REQUIRE( cudaFree(y) == cudaSuccess);
    REQUIRE( cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("cublasSgemv test 3.", "[cublas]") {
    INFO("in_channels > out_channels, with height");

    cublasHandle_t handle;
    REQUIRE( cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);

    const size_t batch_size = 1, n_in_channels = 5, height = 2, width = 4;
    const size_t n_out_channels = 3;
    const size_t n_rows = n_in_channels, n_cols = n_out_channels;

    float *x = nullptr;
    REQUIRE( cudaMallocManaged(&x, batch_size * n_in_channels * height * width * sizeof(float)) == cudaSuccess );
    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_in_channels; ++ch) {
            for(int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    size_t index = w + h * width + ch * width * height + batch * n_in_channels * height * width;
                    x[index] = 1.0 * w + h * 10.0 + ch * 100.0;
                }
            }
        }
    }

    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_in_channels; ++ch) {
            std::cout << "batch=" << batch << ", ch=" << ch << std::endl;
            for(int h = 0; h < height; h++) {
                for (int w = 0; w < width; ++w) {
                    std::cout << x[w + h * width + ch * height * width] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    float *A = nullptr;
    REQUIRE( cudaMallocManaged(&A, n_rows * n_cols * sizeof(float)) == cudaSuccess );
    for(int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < n_cols; ++col) {
            A[col + row * n_cols] = 1.0;
        }
    }

    float *y = nullptr;
    REQUIRE( cudaMallocManaged(&y, batch_size * n_out_channels * height * width * sizeof(float)) == cudaSuccess );
    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_out_channels; ++ch) {
            for(int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    size_t index = w + h * width + ch * width * height + batch * n_out_channels * height * width;
                    y[index] = 0.0;
                }
            }
        }
    }

    const float alpha = 1.0, beta = 0.0;
    for(int batch = 0; batch < batch_size; batch++) {
        for(int h = 0; h < height; h++) {
            for(int w = 0; w < width; w ++) {
                float *xx = x + w + h * width + batch * n_in_channels * height * width;
                float *yy = y + w + h * width + batch * n_out_channels * height * width;
                REQUIRE( cublasSgemv_v2(handle, CUBLAS_OP_T,
                                    n_rows, n_cols, &alpha, A, n_rows,
                                    xx, width * height, &beta, yy, width * height) == CUBLAS_STATUS_SUCCESS );
            }
        }
    }
    cudaDeviceSynchronize();

    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_out_channels; ++ch) {
            std::cout << "batch=" << batch << ", ch=" << ch << std::endl;
            for(int h = 0; h < height; h++) {
                for (int w = 0; w < width; ++w) {
                    std::cout << y[w + h * width + ch * height * width] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    float pat[height][width] = {
        {1000, 1005.0, 1010.0, 1015.0},
        {1050, 1055.0, 1060.0, 1065.0},
    };
    for(int batch = 0; batch < batch_size; ++batch) {
        for(int ch = 0; ch < n_out_channels; ++ch) {
            for(int h = 0; h < height; h++) {
                for (int w = 0; w < width; ++w) {
                    size_t index = w + h * width + ch * height * width + batch * n_out_channels * height * width;
                    CHECK(y[index] == Approx(pat[h][w]));
                }
            }
        }
    }

    REQUIRE( cudaFree(A) == cudaSuccess);
    REQUIRE( cudaFree(x) == cudaSuccess);
    REQUIRE( cudaFree(y) == cudaSuccess);
    REQUIRE( cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS);
}
