#include <cmath>
#include <iostream>
#include <iomanip>      // std::setprecision
#include "catch.hpp"
#include "tensor.h"
#include "H5Cpp.h"

using namespace H5;

void LoadHidden(size_t i, int receptive_field, int upsample_factor, hsize_t *lc_dims,
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

    hsize_t offset[2] = { static_cast<hsize_t>(read_offset), 0 };          // hyperslab offset in the file
    hsize_t count[2]  = { read_count + 1, lc_dims[1] };  // size of the hyperslab in the file
    lc_dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );

    hsize_t dimsm[2] = { 16, lc_dims[1] };           // memory space dimensions
    DataSpace memspace( 2, dimsm );

    hsize_t offset_out[2] = { 0, 0 };                    // hyperslab offset in memory
    hsize_t count_out[2]  = { read_count + 1, lc_dims[1] };  // size of the hyperslab in memory
    memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );

    float data_out[16][80] = { 0 };
    lc_datasset.read( &data_out[0][0], PredType::NATIVE_FLOAT, memspace, lc_dataspace );

    for(int ch = 0; ch < lc_dims[1]; ch++) {
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

TEST_CASE("hidden state fill test", "[hidden]") {
    const int upsample_factor = 160;
    const int receptive_field = 2048;

    const H5std_string FILE_NAME( "/tmp/lc_file.h5" );
    const H5std_string DATASET_NAME( "local_condition" );
    H5File file( FILE_NAME, H5F_ACC_RDONLY );
    auto lc_datasset = file.openDataSet( DATASET_NAME );

    auto lc_dataspace = lc_datasset.getSpace();
    hsize_t lc_dims[2];
    int ndims_lc = lc_dataspace.getSimpleExtentDims( lc_dims, NULL);

    cudnn::Tensor4d h_tensor(1, lc_dims[1], 1, receptive_field);
    auto h_data = h_tensor.CreateArray4f32();
    h_data.InitializeWithZeros();

    SECTION("i = 0") {
        const size_t i = 0;
        LoadHidden(i, receptive_field, upsample_factor, lc_dims, lc_datasset, lc_dataspace, h_tensor, h_data);

        CHECK(h_data(0, 0, 0, 0) == Approx(0.0));
        CHECK(h_data(0, 0, 0, 2046) == Approx(0.0));
        CHECK(h_data(0, 0, 0, 2047) == Approx(0.086247474));
        CHECK(h_data(0, 1, 0, 2046) == Approx(0.0));
        CHECK(h_data(0, 1, 0, 2047) == Approx(0.14451377));
    }

    SECTION("i = 2") {
        const size_t i = 2;
        LoadHidden(i, receptive_field, upsample_factor, lc_dims, lc_datasset, lc_dataspace, h_tensor, h_data);

        CHECK(h_data(0, 0, 0, 0) == Approx(0.0));
        CHECK(h_data(0, 0, 0, 2046) == Approx(0.086247474));
        CHECK(h_data(0, 0, 0, 2047) == Approx(0.086247474));
        CHECK(h_data(0, 1, 0, 2046) == Approx(0.14451377));
        CHECK(h_data(0, 1, 0, 2047) == Approx(0.14451377));
    }

    SECTION("i = receptive_field - 1") {
        const size_t i = receptive_field - 1;
        LoadHidden(i, receptive_field, upsample_factor, lc_dims, lc_datasset, lc_dataspace, h_tensor, h_data);

        CHECK(h_data(0, 0, 0, 0) == Approx(0.086247474));
        CHECK(h_data(0, 0, 0, 2046) == Approx(0.20645098));
        CHECK(h_data(0, 0, 0, 2047) == Approx(0.20645098));
        CHECK(h_data(0, 1, 0, 2046) == Approx(0.14506266));
        CHECK(h_data(0, 1, 0, 2047) == Approx(0.14506266));
    }

    SECTION("i = receptive_field * 2 - 1") {
        const size_t i = receptive_field * 2 - 1;
        LoadHidden(i, receptive_field, upsample_factor, lc_dims, lc_datasset, lc_dataspace, h_tensor, h_data);

        CHECK(h_data(0, 0, 0, 0)    == Approx(0.20645098));
        CHECK(h_data(0, 0, 0, 2046) == Approx(0.20357320));
        CHECK(h_data(0, 0, 0, 2047) == Approx(0.20357320));
        CHECK(h_data(0, 1, 0, 2046) == Approx(0.17770557));
        CHECK(h_data(0, 1, 0, 2047) == Approx(0.17770557));
    }

    SECTION("i = receptive_field * 3 + 10") {
        const size_t i = receptive_field * 3 + 10;
        LoadHidden(i, receptive_field, upsample_factor, lc_dims, lc_datasset, lc_dataspace, h_tensor, h_data);

        CHECK(h_data(0, 0, 0, 0)    == Approx(0.2035732));
        CHECK(h_data(0, 0, 0, 1)    == Approx(0.2035732));
        CHECK(h_data(0, 0, 0, 2046) == Approx(0.21821374));
        CHECK(h_data(0, 0, 0, 2047) == Approx(0.21821374));
        CHECK(h_data(0, 1, 0, 2046) == Approx(0.21241343));
        CHECK(h_data(0, 1, 0, 2047) == Approx(0.21241343));
    }
}
