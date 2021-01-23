#pragma once

#include "../types.hpp"
#include "../math.hpp"

#include <algorithm>
#include <type_traits>


namespace border {

struct mirror {
    template<typename T>
    static constexpr auto value(image<T> const& img, index2_t const& i) -> T;
};

template<typename T>
constexpr auto mirror::value(image<T> const& img, index2_t const& i) -> T
{
    index_t const x = i.x >= 0 ? (i.x < img.shape().x ? i.x : 2 * img.shape().x - i.x - 1) : (-1 - i.x);
    index_t const y = i.y >= 0 ? (i.y < img.shape().y ? i.y : 2 * img.shape().y - i.y - 1) : (-1 - i.y);

    return img[{x, y}];
}


struct mirror_x {
    template<typename T>
    static constexpr auto value(image<T> const& img, index2_t const& i) -> T;
};

template<typename T>
constexpr auto mirror_x::value(image<T> const& img, index2_t const& i) -> T
{
    index_t const x = i.x >= 0 ? (i.x < img.shape().x ? i.x : 2 * img.shape().x - i.x - 1) : (-1 - i.x);

    return i.y >= 0 && i.y < img.shape().y ? img[{x, i.y}] : zero<T>();
}


struct mirror_y {
    template<typename T>
    static constexpr auto value(image<T> const& img, index2_t const& i) -> T;
};

template<typename T>
constexpr auto mirror_y::value(image<T> const& img, index2_t const& i) -> T
{
    index_t const y = i.y >= 0 ? (i.y < img.shape().y ? i.y : 2 * img.shape().y - i.y - 1) : (-1 - i.y);

    return i.x >= 0 && i.x < img.shape().x ? img[{i.x, y}] : zero<T>();
}


struct extend {
    template<typename T>
    static constexpr auto value(image<T> const& img, index2_t const& i) -> T;
};

template<typename T>
constexpr auto extend::value(image<T> const& img, index2_t const& i) -> T
{
    index_t const x = std::clamp(i.x, 0, img.shape().x - 1);
    index_t const y = std::clamp(i.y, 0, img.shape().y - 1);

    return img[{x, y}];
}


struct zero {
    template<typename T>
    static constexpr auto value(image<T> const& img, index2_t const& i) -> T;
};

template<typename T>
constexpr auto zero::value(image<T> const& img, index2_t const& i) -> T
{
    return i.x >= 0 && i.x < img.shape().x && i.y >= 0 && i.y < img.shape().y ?
        img[{i.x, i.y}] : ::zero<T>();
}

} /* namespace border */


namespace impl {

template<typename B, typename T, typename S, index_t Nx, index_t Ny>
void conv_generic(image<T>& out, image<T> const& in, kernel<S, Nx, Ny> const& k)
{
    index_t const dx = (Nx - 1) / 2;
    index_t const dy = (Ny - 1) / 2;

    for (index_t cy = 0; cy < in.shape().y; ++cy) {
        for (index_t cx = 0; cx < in.shape().x; ++cx) {
            out[{cx, cy}] = zero<T>();

            for (index_t iy = 0; iy < Ny; ++iy) {
                for (index_t ix = 0; ix < Nx; ++ix) {
                    out[{cx, cy}] += B::value(in, {cx - dx + ix, cy - dy + iy}) * k[{ix, iy}];
                }
            }
        }
    }
}

template<typename T, typename S>
void conv_5x5_extend(image<T>& out, image<T> const& in, kernel<S, 5, 5> const& k);

} /* namespace impl */


#include "convolution.opt.3x3-extend.hpp"
#include "convolution.opt.5x5-extend.hpp"

template<typename B=border::extend, typename T, typename S, index_t Nx, index_t Ny>
void conv(image<T>& out, image<T> const& in, kernel<S, Nx, Ny> const& k)
{
    // workaround for partial function template specialization
    if constexpr (Nx == 5 && Ny == 5 && std::is_same_v<B, border::extend>) {
        impl::conv_5x5_extend<T, S>(out, in, k);
    } else if constexpr (Nx == 3 && Ny == 3 && std::is_same_v<B, border::extend>) {
        impl::conv_3x3_extend<T, S>(out, in, k);
    } else {
        impl::conv_generic<B, T, S, Nx, Ny>(out, in, k);
    }
}
