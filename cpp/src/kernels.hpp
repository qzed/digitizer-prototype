#pragma once

#include "types.hpp"
#include "math.hpp"


namespace kernels {

template<class T>
inline constexpr kernel<T, 3, 3> sobel3_x {
     1,  0, -1,
     2,  0, -2,
     1,  0, -1,
};

template<class T>
inline constexpr kernel<T, 3, 3> sobel3_y {
     1,  2,  1,
     0,  0,  0,
    -1, -2, -1,
};

template<class T>
inline constexpr kernel<T, 3, 3> sobel3_xx {
     1, -2,  1,
     2, -4,  2,
     1, -2,  1,
};

template<class T>
inline constexpr kernel<T, 3, 3> sobel3_yy {
     1,  2,  1,
    -2, -4, -2,
     1,  2,  1,
};

template<class T>
inline constexpr kernel<T, 3, 3> sobel3_xy {
     1,  0, -1,
     0,  0,  0,
    -1,  0,  1,
};


template<class T, index_t Nx, index_t Ny>
auto gaussian(T sigma) -> kernel<T, Nx, Ny>
{
    static_assert(Nx % 2 == 1);
    static_assert(Ny % 2 == 1);

    auto k = kernel<T, Nx, Ny>{};

    T sum = static_cast<T>(0.0);

    for (index_t j = 0; j < Ny; j++) {
        for (index_t i = 0; i < Nx; i++) {
            auto const x = (vec2_t<T> {
                static_cast<T>(i - (Nx - 1) / 2),
                static_cast<T>(j - (Ny - 1) / 2)
            } / sigma).norm_l2();

            auto const v = std::exp(-static_cast<T>(0.5) * x * x);

            k[{i, j}] = v;
            sum += v;
        }
    }

    transform_inplace(k, [&](auto const& x) { return x / sum; });
    return k;
}

} /* namespace kernel */
