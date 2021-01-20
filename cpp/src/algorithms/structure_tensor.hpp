#pragma once

#include "../types.hpp"
#include "../math.hpp"
#include "../kernels.hpp"
#include "convolution.hpp"


template<typename Bx=border::zero, typename By=border::zero, typename T, index Nx=3, index Ny=3>
void structure_tensor_prep(image<mat2s<T>>& out, image<T> const& in,
                           kernel<T, Nx, Ny> const& kx=kernels::sobel3_x,
                           kernel<T, Nx, Ny> const& ky=kernels::sobel3_y)
{
    assert(in.shape() == out.shape());

    index const dx = (Nx - 1) / 2;
    index const dy = (Ny - 1) / 2;

    for (index cy = 0; cy < in.shape().y; ++cy) {
        for (index cx = 0; cx < in.shape().x; ++cx) {
            T gx = zero<T>();
            T gy = zero<T>();

            for (index iy = 0; iy < Ny; ++iy) {
                for (index ix = 0; ix < Nx; ++ix) {
                    gx += Bx::value(in, {cx - dx + ix, cy - dy + iy}) * kx[{ix, iy}];
                    gy += By::value(in, {cx - dx + ix, cy - dy + iy}) * ky[{ix, iy}];
                }
            }

            out[{cx, cy}].xx = gx * gx;
            out[{cx, cy}].xy = gx * gy;
            out[{cx, cy}].yy = gy * gy;
        }
    }
}

#include "structure_tensor.opt.hpp"
