#pragma once

#include "types.hpp"

#include "math/vec2.hpp"
#include "math/mat2.hpp"


struct touch_point {
    f32 scale;
    f32 confidence;
    math::vec2_t<f32>  mean;
    math::mat2s_t<f32> cov;
};
