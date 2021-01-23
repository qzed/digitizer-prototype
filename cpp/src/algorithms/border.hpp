#pragma once

#include "../types.hpp"
#include "../math.hpp"

#include <algorithm>


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
