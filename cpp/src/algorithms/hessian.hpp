#pragma once

#include "../types.hpp"
#include "../math.hpp"
#include "../kernels.hpp"
#include "convolution.hpp"

#include "hessian.opt.zero.hpp"


namespace impl {

template<typename B=border::zero, typename T>
void hessian_generic(image<mat2s<T>>& out, image<T> const& in)
{
    auto const& kxx = kernels::sobel3_xx<T>;
    auto const& kyy = kernels::sobel3_yy<T>;
    auto const& kxy = kernels::sobel3_xy<T>;

    index_t const nx = kxx.shape().x;
    index_t const ny = kxx.shape().y;

    index_t const dx = (nx - 1) / 2;
    index_t const dy = (ny - 1) / 2;

    for (index_t cy = 0; cy < in.shape().y; ++cy) {
        for (index_t cx = 0; cx < in.shape().x; ++cx) {
            T hxx = zero<T>();
            T hxy = zero<T>();
            T hyy = zero<T>();

            for (index_t iy = 0; iy < ny; ++iy) {
                for (index_t ix = 0; ix < nx; ++ix) {
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

} /* namespace impl */


template<typename B=border::zero, typename T>
void hessian(image<mat2s<T>>& out, image<T> const& in)
{
    assert(in.shape() == out.shape());

    if constexpr (std::is_same_v<B, border::zero>) {
        impl::hessian_zero<T>(out, in);
    } else {
        impl::hessian_generic<B, T>(out, in);
    }
}
