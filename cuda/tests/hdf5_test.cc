#include <cstdio>
#include <vector>
#include "catch.hpp"
#include "linear.h"
#include "conv1d.h"
#include "H5Cpp.h"

using namespace H5;

const H5std_string FILE_NAME( "/tmp/model_ckpt.h5" );


TEST_CASE("Load Linear layer and forward.", "[layers::Linear][hdf5]") {
    const H5std_string DATASET_NAME_WEIGHT( "linear/weight" );
    const H5std_string DATASET_NAME_BIAS( "linear/bias" );

    H5File file( FILE_NAME, H5F_ACC_RDONLY );
    DataSet ds_weight = file.openDataSet( DATASET_NAME_WEIGHT );
    DataSet ds_bias = file.openDataSet( DATASET_NAME_BIAS );

    DataSpace dataspace_weight = ds_weight.getSpace();
    hsize_t weight_dims[2];
    int ndims_weight = dataspace_weight.getSimpleExtentDims( weight_dims, NULL);
    REQUIRE(weight_dims[0] == 256);
    REQUIRE(weight_dims[1] == 256);

    DataSpace dataspace_bias = ds_bias.getSpace();
    hsize_t bias_dims[1];
    int ndims_bias = dataspace_bias.getSimpleExtentDims( bias_dims, NULL);
    REQUIRE(bias_dims[0] == 256);

    cudnn::Context context;
    layers::Linear linear(context, 256, 256);

    float *weight_data = new float[linear.size()];
    ds_weight.read(weight_data, PredType::NATIVE_FLOAT);
    ds_bias.read(linear.bias_data.data(), PredType::NATIVE_FLOAT);

    linear.weight(weight_data);
    delete[] weight_data;

    cudnn::Tensor4d input_tensor(1, 256, 1, 3);
    auto input_data = input_tensor.CreateArray4f32();
    for(int ch = 0; ch < input_tensor.n_channels; ch ++) {
        for(int w = 0; w < input_tensor.width; w ++) {
            input_data(0, ch, 0, w) = w * 1.0;
        }
    }

    auto output_tensor = linear.CreateOutputTensor(input_tensor);
    auto output_data = output_tensor.CreateArray4f32();
    output_data.InitializeWithZeros();

    linear.Forward(input_tensor, input_data, output_tensor, output_data);
    cudaDeviceSynchronize();

    CHECK( output_data(0, 0, 0, 0) == Approx(-0.1494).epsilon(0.001) );
    CHECK( output_data(0, 0, 0, 1) == Approx(-25.5964).epsilon(0.001) );
    CHECK( output_data(0, 1, 0, 0) == Approx(-0.1569).epsilon(0.001) );
    CHECK( output_data(0, 2, 0, 0) == Approx(-0.2280).epsilon(0.001) );
}

TEST_CASE("Load Conv1D layer and forward.", "[layers::Conv1D][hdf5]") {
    const H5std_string DATASET_NAME_WEIGHT( "layers/0/x_l_conv/weight" );
    const H5std_string DATASET_NAME_BIAS( "layers/0/x_l_conv/bias" );

    H5File file( FILE_NAME, H5F_ACC_RDONLY );
    DataSet ds_weight = file.openDataSet( DATASET_NAME_WEIGHT );
    DataSet ds_bias = file.openDataSet( DATASET_NAME_BIAS );

    DataSpace dataspace_weight = ds_weight.getSpace();
    hsize_t weight_dims[3];
    int ndims_weight = dataspace_weight.getSimpleExtentDims( weight_dims, NULL);
    REQUIRE(weight_dims[0] == 256);
    REQUIRE(weight_dims[1] == 1);
    REQUIRE(weight_dims[2] == 1);

    DataSpace dataspace_bias = ds_bias.getSpace();
    hsize_t bias_dims[1];
    int ndims_bias = dataspace_bias.getSimpleExtentDims( bias_dims, NULL);
    REQUIRE(bias_dims[0] == 256);

    cudnn::Context context;
    layers::Conv1D conv1d(context, 1, 256, 1);

    ds_weight.read(conv1d.weight_data.data(), PredType::NATIVE_FLOAT);
    ds_bias.read(conv1d.bias_data.data(), PredType::NATIVE_FLOAT);

    cudnn::Tensor4d input_tensor(1, 1, 1, 10);
    auto input_data = input_tensor.CreateArray4f32();
    for(int ch = 0; ch < input_tensor.n_channels; ch ++) {
        for(int w = 0; w < input_tensor.width; w ++) {
            input_data(0, ch, 0, w) = 2.0;
        }
    }

    auto output_tensor = conv1d.CreateOutputTensor(input_tensor);
    auto output_data = output_tensor.CreateArray4f32();
    output_data.InitializeWithZeros();

    conv1d.Forward(input_tensor, input_data, output_tensor, output_data);
    cudaDeviceSynchronize();

    CHECK( output_data(0, 0, 0, 0) == Approx(0.9476).epsilon(0.001) );
    CHECK( output_data(0, 0, 0, 1) == Approx(0.9476).epsilon(0.001) );
    CHECK( output_data(0, 1, 0, 0) == Approx(0.2439).epsilon(0.001) );
    CHECK( output_data(0, 2, 0, 0) == Approx(-1.2339).epsilon(0.001) );
}

TEST_CASE("Write to file test.", "[hdf5]") {
    std::vector<float> samples(200, 1.0);

    H5File ofile( "/tmp/samples.h5", H5F_ACC_TRUNC );

    float fillvalue = 0; // Fill value for the dataset
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_FLOAT, &fillvalue);

    hsize_t fdim[1] = { 0 };
    fdim[0] = samples.size();
    DataSpace fspace( 1, fdim );
    fspace.selectAll();

    auto dataset = ofile.createDataSet("samples", PredType::NATIVE_FLOAT, fspace, plist);

    hsize_t mdim[1] = { 0 };
    mdim[0] = samples.size();
    DataSpace mspace( 1, mdim );

    dataset.write(samples.data(), PredType::NATIVE_FLOAT, mspace, fspace);
    dataset.close();
    ofile.close();
}