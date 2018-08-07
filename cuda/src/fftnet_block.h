#pragma once
#ifndef __FFTNET_BLOCK_H__
#define __FFTNET_BLOCK_H__

#include "cudnn_context.h"
#include "conv1d.h"
#include "relu.h"

class FFTNetBlock {
    const cudnn::Context &_context;
    cudnn::ReLU _relu;

    int _local_condition_channels;
    int _shift;

    void _SplitXByShift(const cudnn::Tensor4d &tensor, const cudnn::Array4f32 &input_data,
                        cudnn::Array4f32 &out_l_data, cudnn::Array4f32 &out_r_data) const;
    void _SplitHByShift(const cudnn::Tensor4d &tensor, const cudnn::Array4f32 &input_data,
                        int width,
                        cudnn::Array4f32 &out_l_data,
                        cudnn::Array4f32 &out_r_data);

    void _AddTensor(const cudnn::Tensor4d &tensor,
                    const cudnn::Array4f32 &data,
                    cudnn::Array4f32 &output);
public:
    layers::Conv1D x_l_conv1d;
    layers::Conv1D x_r_conv1d;
    layers::Conv1D *h_l_conv1d;
    layers::Conv1D *h_r_conv1d;
    layers::Conv1D out_conv1d;

public:
    FFTNetBlock(const cudnn::Context &context,
                int in_channels, 
                int out_channels, 
                int shift,
                int local_condition_channels=-1);
    ~FFTNetBlock();

    FFTNetBlock(const FFTNetBlock& other) = delete;
    FFTNetBlock(FFTNetBlock&& other) = delete;
    FFTNetBlock& operator=(const FFTNetBlock& other)  = delete;
    FFTNetBlock& operator=(FFTNetBlock&& other) = delete;

    cudnn::Tensor4d CreateOutputTensor(const cudnn::Tensor4d &input_tensor);
    void Forward(const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
                    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data);
    void Forward(const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
                    const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
                    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data);
};

#endif // __FFTNET_BLOCK_H__