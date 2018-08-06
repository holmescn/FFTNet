#pragma once
#ifndef __CONV1D_H__
#define __CONV1D_H__

#include "cudnn_context.h"
#include "kernel.h"
#include "tensor.h"
#include "convolution.h"

namespace layers {
    class Conv1d {
        const cudnn::Context &_context;
        cudnn::Kernel _weights;
        cudnn::Convolution _convolution;
    public:
        Conv1d(const cudnn::Context &context,
                int in_channels, int out_channels, int kernel_size, int stride = 1,
                int padding=0, bool bias=false);
        ~Conv1d();
        Conv1d(const Conv1d& other) = delete;
        Conv1d(Conv1d&& other) = delete;
        Conv1d& operator=(const Conv1d& other) = delete;
        Conv1d& operator=(Conv1d&& other) = delete;

        cudnn::Array4f32 CreateWeightsArray4f32();
        
        void operator()(const cudnn::Tensor4d &input_tensor, const cudnn::Array4f32 &input_data,
                        const cudnn::Tensor4d &output_tensor, const cudnn::Array4f32 &output_data,
                        const cudnn::Array4f32 &weights_data);
    };
}

#endif // __CONV1D_H__