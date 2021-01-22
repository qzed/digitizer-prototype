#pragma once

#include "../types.hpp"
#include "../math.hpp"
#include "../kernels.hpp"
#include "convolution.hpp"


template<typename B=border::zero, typename T>
void hessian(image<mat2s<T>>& out, image<T> const& in)
{
    assert(in.shape() == out.shape());

    auto const& kxx = kernels::sobel3_xx<T>;
    auto const& kyy = kernels::sobel3_yy<T>;
    auto const& kxy = kernels::sobel3_xy<T>;

    index const nx = kxx.shape().x;
    index const ny = kxx.shape().y;

    index const dx = (nx - 1) / 2;
    index const dy = (ny - 1) / 2;

    for (index cy = 0; cy < in.shape().y; ++cy) {
        for (index cx = 0; cx < in.shape().x; ++cx) {
            T hxx = zero<T>();
            T hxy = zero<T>();
            T hyy = zero<T>();

            for (index iy = 0; iy < ny; ++iy) {
                for (index ix = 0; ix < nx; ++ix) {
                    hxx += B::value(in, {cx - dx + ix, cy - dy + iy}) * kxx[{ix, iy}];
                    hxy += B::value(in, {cx - dx + ix, cy - dy + iy}) * kxy[{ix, iy}];
                    hyy += B::value(in, {cx - dx + ix, cy - dy + iy}) * kyy[{ix, iy}];
                }
            }

            out[{cx, cy}].xx = hxx;
            out[{cx, cy}].xy = hxy;
            out[{cx, cy}].yy = hyy;
        }
    }
}

#include "hessian.opt.hpp"
