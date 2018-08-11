#include <cmath>
#include <iostream>
#include <iomanip>      // std::setprecision
#include "catch.hpp"
#include "tensor.h"
#include "H5Cpp.h"

using namespace H5;

void LoadLocalCondition(int receptive_field, int upsample_factor,
    const DataSet& lc_dataset,
    hsize_t lc_dims[2],
    cudnn::Array4f32 &local_condition)
{
    cudnn::Array4f32 raw(1, 1, lc_dims[0], lc_dims[1]);
    lc_dataset.read(raw.data(), H5::PredType::NATIVE_FLOAT);

    for(int ch = 0; ch < local_condition.dim(2); ch++) {
        for(int w = receptive_field; w < local_condition.dim(3); w++) {
            int ww = (w - receptive_field) / upsample_factor;
            local_condition(0, 0, ch, w) = raw(0, 0, ww, ch);
            // std::cout << "local_condition(0, 0, " << ch << ", " << w << ") = " << local_condition(0, 0, ch, w) << std::endl;
            // std::cout << "raw(0, 0, " << ww << ", " << ch << ") = " << raw(0, 0, ww, ch) << std::endl;
        }
    }
}

TEST_CASE("Load local condition test", "[local_condition]") {
    const int upsample_factor = 160;
    const int receptive_field = 2048;

    const H5std_string FILE_NAME( "/tmp/lc_file.h5" );
    const H5std_string DATASET_NAME( "local_condition" );
    H5File file( FILE_NAME, H5F_ACC_RDONLY );
    auto lc_dataset = file.openDataSet( DATASET_NAME );

    auto lc_dataspace = lc_dataset.getSpace();
    hsize_t lc_dims[2];
    int ndims_lc = lc_dataspace.getSimpleExtentDims( lc_dims, NULL);

    cudnn::Array4f32 local_condition(1, 1, lc_dims[1], lc_dims[0] * upsample_factor + receptive_field);
    local_condition.InitializeWithZeros();
    LoadLocalCondition(receptive_field, upsample_factor, lc_dataset, lc_dims, local_condition);

    cudnn::Tensor4d h_tensor(1, lc_dims[1], 1, receptive_field);
    auto h_data = h_tensor.CreateArray4f32();
    h_data.InitializeWithZeros();

    SECTION("local_condition") {
        CHECK(local_condition(0, 0, 0,    0) == Approx(0.0));
        CHECK(local_condition(0, 0, 0, 2047) == Approx(0.0));
        CHECK(local_condition(0, 0, 0, 2048) == Approx(0.086247474));
        CHECK(local_condition(0, 0, 0, 55807) == Approx(0.1049664));

        CHECK(local_condition(0, 0, 1,    0) == Approx(0.0));
        CHECK(local_condition(0, 0, 1, 2047) == Approx(0.0));
        CHECK(local_condition(0, 0, 1, 2048) == Approx(0.14451377)); 
        CHECK(local_condition(0, 0, 1, 55807) == Approx(0.11696948));
    }

    SECTION("i = 0") {
        const size_t i = 0;
        for(int ch = 0; ch < h_tensor.n_channels; ch++) {
            for(int w = 0; w < h_tensor.width; w++) {
                h_data(0, ch, 0, w) = local_condition(0, 0, ch, w + i + 1);
            }
        }

        CHECK(h_data(0, 0, 0, 0) == Approx(0.0));
        CHECK(h_data(0, 0, 0, 2046) == Approx(0.0));
        CHECK(h_data(0, 0, 0, 2047) == Approx(0.086247474));
        CHECK(h_data(0, 1, 0, 2046) == Approx(0.0));
        CHECK(h_data(0, 1, 0, 2047) == Approx(0.14451377));
    }

    SECTION("i = 2") {
        const size_t i = 2;
        for(int ch = 0; ch < h_tensor.n_channels; ch++) {
            for(int w = 0; w < h_tensor.width; w++) {
                h_data(0, ch, 0, w) = local_condition(0, 0, ch, w + i + 1);
            }
        }

        CHECK(h_data(0, 0, 0, 0) == Approx(0.0));
        CHECK(h_data(0, 0, 0, 2046) == Approx(0.086247474));
        CHECK(h_data(0, 0, 0, 2047) == Approx(0.086247474));
        CHECK(h_data(0, 1, 0, 2046) == Approx(0.14451377));
        CHECK(h_data(0, 1, 0, 2047) == Approx(0.14451377));
    }

    SECTION("i = receptive_field - 1") {
        const size_t i = receptive_field - 1;
        for(int ch = 0; ch < h_tensor.n_channels; ch++) {
            for(int w = 0; w < h_tensor.width; w++) {
                h_data(0, ch, 0, w) = local_condition(0, 0, ch, w + i + 1);
            }
        }

        CHECK(h_data(0, 0, 0, 0) == Approx(0.086247474));
        CHECK(h_data(0, 0, 0, 2046) == Approx(0.20645098));
        CHECK(h_data(0, 0, 0, 2047) == Approx(0.20645098));
        CHECK(h_data(0, 1, 0, 2046) == Approx(0.14506266));
        CHECK(h_data(0, 1, 0, 2047) == Approx(0.14506266));
    }

    SECTION("i = receptive_field * 2 - 1") {
        const size_t i = receptive_field * 2 - 1;
        for(int ch = 0; ch < h_tensor.n_channels; ch++) {
            for(int w = 0; w < h_tensor.width; w++) {
                h_data(0, ch, 0, w) = local_condition(0, 0, ch, w + i + 1);
            }
        }

        CHECK(h_data(0, 0, 0, 0)    == Approx(0.20645098));
        CHECK(h_data(0, 0, 0, 2046) == Approx(0.20357320));
        CHECK(h_data(0, 0, 0, 2047) == Approx(0.20357320));
        CHECK(h_data(0, 1, 0, 2046) == Approx(0.17770557));
        CHECK(h_data(0, 1, 0, 2047) == Approx(0.17770557));
    }

    SECTION("i = receptive_field * 3 + 10") {
        const size_t i = receptive_field * 3 + 10;
        for(int ch = 0; ch < h_tensor.n_channels; ch++) {
            for(int w = 0; w < h_tensor.width; w++) {
                h_data(0, ch, 0, w) = local_condition(0, 0, ch, w + i + 1);
            }
        }

        CHECK(h_data(0, 0, 0, 0)    == Approx(0.2035732));
        CHECK(h_data(0, 0, 0, 1)    == Approx(0.2035732));
        CHECK(h_data(0, 0, 0, 2046) == Approx(0.21821374));
        CHECK(h_data(0, 0, 0, 2047) == Approx(0.21821374));
        CHECK(h_data(0, 1, 0, 2046) == Approx(0.21241343));
        CHECK(h_data(0, 1, 0, 2047) == Approx(0.21241343));
    }
}
