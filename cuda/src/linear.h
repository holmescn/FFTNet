#pragma once
#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "cudnn_context.h"
#include "tensor.h"

namespace layers {
    class Linear {
        const cudnn::Context &_context;

    public:
        cudnn::Array4f32 weight_data;
        cudnn::Array4f32 bias_data;

    public:
        Linear(const cudnn::Context &context,
                int in_features,
                int out_features,
                bool use_bias=true);
        ~Linear();
        Linear(const Linear& other) = delete;
        Linear(Linear&& other) = delete;
        Linear& operator=(const Linear& other) = delete;
        Linear& operator=(Linear&& other) = delete;

        void Forward(const cudnn::Tensor4d &input_tensor,
                     const cudnn::Array4f32 &input_data,
                     const cudnn::Tensor4d &output_tensor,
                     cudnn::Array4f32 &output_data);

        cudnn::Tensor4d CreateOutputTensor(const cudnn::Tensor4d &input_tensor);
    };
}

#endif // __LINEAR_H__