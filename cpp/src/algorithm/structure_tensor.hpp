#pragma once

#include "types.hpp"
#include "math.hpp"
#include "kernels.hpp"

#include "algorithm/border.hpp"

#include "math/num.hpp"
#include "math/mat2.hpp"

#include "algorithm/structure_tensor.opt.3x3-zero.hpp"


namespace impl {

template<typename Bx, typename By, typename T, index_t Nx, index_t Ny>
void structure_tensor_generic(image<math::mat2s_t<T>>& out, image<T> const& in,
                              kernel<T, Nx, Ny> const& kx, kernel<T, Nx, Ny> const& ky)
{
    index_t const dx = (Nx - 1) / 2;
    index_t const dy = (Ny - 1) / 2;

    for (index_t cy = 0; cy < in.shape().y; ++cy) {
        for (index_t cx = 0; cx < in.shape().x; ++cx) {
            T gx = math::num<T>::zero;
            T gy = math::num<T>::zero;

            for (index_t iy = 0; iy < Ny; ++iy) {
                for (index_t ix = 0; ix < Nx; ++ix) {
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

} /* namespace impl */


template<typename Bx=border::zero, typename By=border::zero, typename T, index_t Nx=3, index_t Ny=3>
void structure_tensor(image<math::mat2s_t<T>>& out, image<T> const& in,
                      kernel<T, Nx, Ny> const& kx=kernels::sobel3_x<T>,
                      kernel<T, Nx, Ny> const& ky=kernels::sobel3_y<T>)
{
    assert(in.shape() == out.shape());

    // workaround for partial function template specialization
    if constexpr (Nx == 3 && Ny == 3 && std::is_same_v<Bx, border::zero> && std::is_same_v<By, border::zero>) {
        impl::structure_tensor_3x3_zero<T>(out, in, kx, ky);
    } else {
        impl::structure_tensor_generic<Bx, By, T, Nx, Ny>(out, in, kx, ky);
    }
}
