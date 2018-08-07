#include "fftnet_block.h"
#include "exception.h"

FFTNetBlock::FFTNetBlock(
    const cudnn::Context &context,
    int in_channels,
    int out_channels,
    int shift,
    int local_condition_channels)
: _context(context),
  _x_kernel(in_channels, out_channels, 1, 1),
  _x_conv(context, _x_kernel),
  _x_l_weight(_x_kernel.CreateArray4f32()),
  _x_l_bias(_x_conv.CreateBiasArray4f32()),
  _x_r_weight(_x_kernel.CreateArray4f32()),
  _x_r_bias(_x_conv.CreateBiasArray4f32()),
  _out_kernel(out_channels, out_channels, 1, 1),
  _out_conv(context, _x_kernel),
  _out_weight(_out_kernel.CreateArray4f32()),
  _out_bias(_out_conv.CreateBiasArray4f32()),
  _h_kernel(nullptr),
  _h_conv(nullptr),
  _h_l_weight(nullptr),
  _h_l_bias(nullptr),
  _h_r_weight(nullptr),
  _h_r_bias(nullptr),
  _local_condition_channels(local_condition_channels),
  _shift(shift)
{
    _x_l_weight.InitializeWithZeros();
    _x_l_bias.InitializeWithZeros();
    _x_r_weight.InitializeWithZeros();
    _x_r_bias.InitializeWithZeros();

    _out_weight.InitializeWithZeros();
    _out_bias.InitializeWithZeros();

    assert_cudnn_success( cudnnCreateActivationDescriptor(&_activation_descriptor) );
    assert_cudnn_success( cudnnSetActivationDescriptor(_activation_descriptor,
        CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0
    ) );


    if (local_condition_channels > 0) {
        _h_kernel = new cudnn::Kernel(local_condition_channels, out_channels, 1, 1);
        _h_conv = new cudnn::Convolution(context, *_h_kernel);
    }
}

FFTNetBlock::~FFTNetBlock()
{
    assert_cudnn_success( cudnnDestroyActivationDescriptor(_activation_descriptor) );

    if (_h_kernel != nullptr) {
        delete _h_conv;
        delete _h_kernel;
    }
}

cudnn::Tensor4d FFTNetBlock::GetOutputTensor(const cudnn::Tensor4d &tensor)
{
    return cudnn::Tensor4d(tensor.batch_size, tensor.n_channels,
                           tensor.height, tensor.width - this->_shift);
}

void FFTNetBlock::Forward(
    const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data)
{
    // x_l = self.x_l_conv(x[:, :, :-self.shift]) 
    // x_r = self.x_r_conv(x[:, :, self.shift:]) 
    cudnn::Tensor4d x_shift_tensor(x_tensor.batch_size, x_tensor.n_channels, 1, x_tensor.width - this->_shift);
    auto x_l_data = x_shift_tensor.CreateArray4f32();
    auto x_r_data = x_shift_tensor.CreateArray4f32();
    _SplitByShift(x_tensor, x_data, x_l_data, x_r_data);

    auto x_l_out = out_tensor.CreateArray4f32();
    cudnn::Array4f32 &x_r_out = out_data;
    x_l_out = 0.0;
    x_r_out = 0.0;

    _x_conv.Forward(out_tensor, x_l_data, out_tensor, x_l_out, _x_l_weight);
    _x_conv.BiasAdd(_x_l_bias, out_tensor, x_l_out);

    _x_conv.Forward(out_tensor, x_r_data, out_tensor, out_data, _x_r_weight);
    _x_conv.BiasAdd(_x_r_bias, out_tensor, out_data);

    // z = F.relu(x_l + x_r)
    _AddTensor(out_tensor, x_r_out, x_l_out);
    _ReLU(out_tensor, x_l_data, out_data);

    // output = F.relu(self.output_conv(z))
    _out_conv.Forward(out_tensor, out_data, out_tensor, x_l_out, _out_weight);
    _out_conv.BiasAdd(_out_bias, out_tensor, x_l_out);
    _ReLU(out_tensor, x_l_data, out_data);
}

void FFTNetBlock::Forward(
    const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
    const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data)
{
    
}

void FFTNetBlock::_SplitByShift(
    const cudnn::Tensor4d &tensor, const cudnn::Array4f32 &input_data,
    cudnn::Array4f32 &out_l, cudnn::Array4f32 &out_r) const
{
    for(int batch = 0; batch < tensor.batch_size; ++batch) {
        for (int ch = 0; ch < tensor.n_channels; ++ch) {
            for (int row = 0; row < tensor.height; ++row) {
                for (int col = 0; col < tensor.width - this->_shift; ++col) {
                    out_l(batch, ch, row, col) = input_data(batch, ch, row, col);
                }
                for (int col = this->_shift; col < tensor.width; ++col) {
                    out_r(batch, ch, row, col - this->_shift) = input_data(batch, ch, row, col);                    
                }
            }
        }
    }
}

void FFTNetBlock::_AddTensor(
    const cudnn::Tensor4d &tensor,
    const cudnn::Array4f32 &data,
    cudnn::Array4f32 &output)
{
    const float alpha = 1.0, beta = 1.0;
    assert_cudnn_success( cudnnAddTensor(
        static_cast<cudnnHandle_t>(_context),
        &alpha,
        static_cast<cudnnTensorDescriptor_t>(tensor),
        data.data(),
        &beta,
        static_cast<cudnnTensorDescriptor_t>(tensor),
        output.data()) );
    cudaDeviceSynchronize();
}

void FFTNetBlock::_ReLU(
    const cudnn::Tensor4d &tensor,
    const cudnn::Array4f32 &data,
    cudnn::Array4f32 &output)
{
    const float alpha = 1.0, beta = 1.0;
    assert_cudnn_success( cudnnActivationForward(
        static_cast<cudnnHandle_t>(_context),
        _activation_descriptor,
        &alpha,
        static_cast<cudnnTensorDescriptor_t>(tensor),
        data.data(),
        &beta,
        static_cast<cudnnTensorDescriptor_t>(tensor),
        output.data()) );
    cudaDeviceSynchronize();
}
