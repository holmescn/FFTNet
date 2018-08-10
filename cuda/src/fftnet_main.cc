#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include "fftnet.h"
#include "hparams.h"
#include "H5Cpp.h"

using namespace H5;

void LoadH(size_t i, int receptive_field, int upsample_factor,
    const DataSet &lc_datasset,
    const DataSpace &lc_dataspace,
    const cudnn::Tensor4d &h_tensor,
    cudnn::Array4f32 &h_data)
{
    hsize_t read_count = ceil(receptive_field * 1.0 / upsample_factor);
    int read_offset = 0;
    int ii = i - receptive_field;
    if (i > receptive_field) {
        read_offset = floor((ii + 1) * 1.0 / upsample_factor);
    }

    hsize_t offset[2] = { static_cast<hsize_t>(read_offset), 0 };   // hyperslab offset in the file
    hsize_t count[2]  = { read_count + 1, 0 };                      // size of the hyperslab in the file
    count[1] = h_tensor.n_channels;
    lc_dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );

    hsize_t dimsm[2] = { 16, 0 };                                   // memory space dimensions
    dimsm[1] = h_tensor.n_channels;
    DataSpace memspace( 2, dimsm );

    hsize_t offset_out[2] = { 0, 0 };                // hyperslab offset in memory
    hsize_t count_out[2]  = { read_count + 1, 0 };   // size of the hyperslab in memory
    count_out[1] = h_tensor.n_channels;
    memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );

    float data_out[16][80] = { 0 };
    lc_datasset.read( &data_out[0][0], PredType::NATIVE_FLOAT, memspace, lc_dataspace );

    for(int ch = 0; ch < h_tensor.n_channels; ch++) {
        int cnt = ii % upsample_factor, idx = 0;
        for (int w = 0; w < receptive_field; w++) {
            if (i + 1 + w < receptive_field) {
                h_data(0, ch, 0, w) = 0.0;
            } else {
                h_data(0, ch, 0, w) = data_out[idx][ch];
                if (++cnt >= upsample_factor) {
                    cnt = 0;
                    idx += 1;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " <checkpoint_file> "
                  << " <lc_file> "
                  << " <output_file> "
                  << std::endl;
        return 1;
    }

    HParams hparams;
    int upsample_factor = hparams.frame_shift_ms / 1000.0 * hparams.sample_rate;

    // ONLY melspc SUPPORTTED
    int lc_channel =  hparams.num_mels;
    FFTNet fftnet(hparams.n_stacks,
                  hparams.fft_channels,
                  hparams.quantization_channels,
                  lc_channel);

    fftnet.InitializeWithHDF5(argv[1]);

    H5File file( argv[2], H5F_ACC_RDONLY );
    const H5std_string DATASET_NAME( "local_condition" );
    auto ds_local_condition = file.openDataSet( DATASET_NAME );

    auto dataspace_lc = ds_local_condition.getSpace();
    hsize_t lc_dims[2];
    int ndims_lc = dataspace_lc.getSimpleExtentDims( lc_dims, NULL);


    auto start = std::chrono::system_clock::now();
    std::vector<float> samples(fftnet.receptive_field, 0.0);

    cudnn::Tensor4d x_tensor(1, 1, 1, fftnet.receptive_field);
    auto x_data = x_tensor.CreateArray4f32();
    x_data.InitializeWithZeros();

    cudnn::Tensor4d h_tensor(1, lc_dims[1], 1, fftnet.receptive_field);
    auto h_data = h_tensor.CreateArray4f32();
    h_data.InitializeWithZeros();

    auto output_tensor = fftnet.CreateOutputTensor(x_tensor);
    auto output_data = output_tensor.CreateArray4f32();
    output_data.InitializeWithZeros();

    int max_steps = lc_dims[0] * upsample_factor + fftnet.receptive_field;
    for(int i = 0; i < max_steps; i++) {
        auto iter = samples.rbegin();
        for(int w = 0; w < fftnet.receptive_field; w++, iter++) {
            x_data(0, 0, 0, fftnet.receptive_field - w - 1) = *iter;
        }
        LoadH(i, fftnet.receptive_field, upsample_factor, ds_local_condition, dataspace_lc, h_tensor, h_data);
        fftnet.Forward(x_tensor, x_data, h_tensor, h_data, output_tensor, output_data);

        std::cout << "step (" << i << "/" << max_steps << ")\r";
    }
    std::cout << std::endl;

    auto now = std::chrono::system_clock::now();
    auto dt_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start);
    std::cout << dt_in_seconds.count() << std::endl;

    return 0;
}