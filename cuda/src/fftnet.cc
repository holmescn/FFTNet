#include "fftnet.h"

FFTNet::FFTNet(
    int n_stacks,
    int fft_channels, 
    int quantization_channels, 
    int local_condition_channels)
: _context(),
  _n_stacks(n_stacks),
  _fft_channels(fft_channels),
  _quantization_channels(quantization_channels),
  _local_condition_channels(local_condition_channels)
{
    // self.window_shifts = [2 ** i for i in range(self.n_stacks)]
    // self.receptive_field = sum(self.window_shifts) + 1
    _receptive_field = 1;
    for(int i = 0; i < n_stacks; i++) {
        int shift = 1 << i;
        _window_shifts.push_back(shift);
        _receptive_field += shift;
    }

    auto last_shift = _window_shifts.back();
    for(auto iter = _window_shifts.rbegin(); iter != _window_shifts.rend(); iter ++) {
        int shift = *iter;
        int in_channels = (shift == last_shift ? 1 : fft_channels);
        auto fftnet_block = std::make_shared<FFTNetBlock>(_context, in_channels, fft_channels, shift, local_condition_channels);
        layers.push_back(fftnet_block);
    }

    // self.linear = nn.Linear(fft_channels, quantization_channels)
}

FFTNet::~FFTNet()
{
    
}

void FFTNet::Forward(
    const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
    const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data)
{
    // Suppose the input is channels_first.
    Forward_Impl(layers.begin(), x_tensor, x_data, h_tensor, h_data, out_tensor, out_data);
}

void FFTNet::Forward_Impl(
    std::vector<std::shared_ptr<FFTNetBlock>>::iterator iter,
    const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
    const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data)
{
    if (iter != layers.end()) {
        auto p_layer = *iter;
        auto layer_out_tensor = p_layer->CreateOutputTensor(x_tensor);
        auto layer_out_data = layer_out_tensor.CreateArray4f32();

        p_layer->Forward(x_tensor, x_data, h_tensor, h_data, layer_out_tensor, layer_out_data);
        cudaDeviceSynchronize();

        Forward_Impl(std::next(iter), layer_out_tensor, layer_out_data,
                          h_tensor, h_data, out_tensor, out_data);
    } else {
        // output = self.linear(output.transpose(1, 2))
        // return output.transpose(1, 2)
    }
}
