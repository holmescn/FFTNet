#include <cassert>
#include <stdexcept>
#include "convolution.h"
#include "exception.h"

cudnn::Convolution::Convolution(const Context &context, const Kernel &kernel,
                    int pad_height, int pad_width,
                    int height_stride, int width_stride,
                    int dilation_height, int dilation_width)
: _workspace(nullptr), _workspace_size(0), _context(context), _kernel(kernel)
{
    assert_cudnn_success( cudnnCreateConvolutionDescriptor(&_convolution_descriptor) );
    assert_cudnn_success( cudnnSetConvolution2dDescriptor(_convolution_descriptor,
        pad_height,
        pad_width,
        height_stride,
        width_stride,
        dilation_height,
        dilation_width,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT) );
}

cudnn::Convolution::~Convolution()
{
    assert_cudnn_success( cudnnDestroyConvolutionDescriptor(_convolution_descriptor) );
    if (_workspace != nullptr) {
        assert_cuda_success( cudaFree(_workspace) );
    }
}

void cudnn::Convolution::_SetConvolutionProperties(int pad_h,
                                                    int pad_w,
                                                    int stride_h,
                                                    int stride_w,
                                                    int dilation_h,
                                                    int dilation_w,
                                                    cudnn::ConvolutionMode mode,
                                                    cudnn::DataType conpute_type)
{
    int pad_h_0 = 0, pad_w_0 = 0, stride_h_0 = 0, stride_w_0 = 0;
    int dilation_h_0 = 0, dilation_w_0 = 0;
    cudnnConvolutionMode_t mode_0;
    cudnnDataType_t compute_type_0;
    assert_cudnn_success( cudnnGetConvolution2dDescriptor(_convolution_descriptor,
        &pad_h_0,
        &pad_w_0,
        &stride_h_0,
        &stride_w_0,
        &dilation_h_0,
        &dilation_w_0,
        &mode_0,
        &compute_type_0) );

    assert_cudnn_success( cudnnSetConvolution2dDescriptor(_convolution_descriptor,
        pad_h > 0 ? pad_h : pad_h_0,
        pad_w > 0 ? pad_w : pad_w_0,
        stride_h > 0 ? stride_h : stride_h_0,
        stride_w > 0 ? stride_w : stride_w_0,
        dilation_h > 0 ? dilation_h : dilation_h_0,
        dilation_w > 0 ? dilation_w : dilation_w_0,
        mode != ConvolutionMode::Invalid ? static_cast<cudnnConvolutionMode_t>(mode) : mode_0,
        conpute_type != DataType::Invalid ? static_cast<cudnnDataType_t>(conpute_type) :
                                            compute_type_0) );
}

void cudnn::Convolution::SetMode(cudnn::ConvolutionMode mode)
{
    _SetConvolutionProperties(0, 0, 0, 0, 0, 0, mode, DataType::Invalid);
}

void cudnn::Convolution::SetPadding(int pad_h, int pad_w)
{
    _SetConvolutionProperties(pad_h, pad_w, 0, 0, 0, 0, ConvolutionMode::Invalid, DataType::Invalid);
}

void cudnn::Convolution::SetStride(int stride_h, int stride_w)
{
    _SetConvolutionProperties(0, 0, stride_h, stride_w, 0, 0, ConvolutionMode::Invalid, DataType::Invalid);
}

void cudnn::Convolution::Forward(
    const cudnn::Tensor4d &input_tensor, const cudnn::Array4f32 &input_data,
    const cudnn::Tensor4d &output_tensor, const cudnn::Array4f32 &output_data,
    const cudnn::Array4f32 &kernel_data)
{
    _PrepareWorkspace(input_tensor, output_tensor);

    const float alpha = 1.0, beta = 0.0;
    assert_cudnn_success( cudnnConvolutionForward(
        static_cast<cudnnHandle_t>(_context),
        &alpha,
        static_cast<cudnnTensorDescriptor_t>(input_tensor),
        input_data.data(),
        static_cast<cudnnFilterDescriptor_t>(_kernel),
        kernel_data.data(),
        static_cast<cudnnConvolutionDescriptor_t>(_convolution_descriptor),
        _convolution_forward_algo,
        _workspace,
        _workspace_size,
        &beta,
        static_cast<cudnnTensorDescriptor_t>(output_tensor),
        output_data.data()) );
    cudaDeviceSynchronize();
}

void cudnn::Convolution::_PrepareWorkspace(const cudnn::Tensor4d &input_tensor,
                                            const cudnn::Tensor4d &output_tensor)
{
    if (_workspace == nullptr) {
        assert_cudnn_success(cudnnGetConvolutionForwardAlgorithm(
            static_cast<cudnnHandle_t>(_context),
            static_cast<cudnnTensorDescriptor_t>(input_tensor),
            static_cast<cudnnFilterDescriptor_t>(_kernel),
            static_cast<cudnnConvolutionDescriptor_t>(_convolution_descriptor),
            static_cast<cudnnTensorDescriptor_t>(output_tensor),
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            /*memoryLimitInBytes=*/0,
            &_convolution_forward_algo));
    }

    size_t workspace_size = 0;
    assert_cudnn_success(cudnnGetConvolutionForwardWorkspaceSize(
        static_cast<cudnnHandle_t>(_context),
        static_cast<cudnnTensorDescriptor_t>(input_tensor),
        static_cast<cudnnFilterDescriptor_t>(_kernel),
        static_cast<cudnnConvolutionDescriptor_t>(_convolution_descriptor),
        static_cast<cudnnTensorDescriptor_t>(output_tensor),
        _convolution_forward_algo,
        &workspace_size));

    if (workspace_size != _workspace_size && _workspace != nullptr) {
        assert_cuda_success( cudaFree(_workspace) );
    }

    _workspace_size = workspace_size;
    assert_cuda_success( cudaMalloc(&_workspace, _workspace_size) );
}
