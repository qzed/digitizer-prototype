#pragma once

#include "types.hpp"


inline constexpr auto ravel(index2_t const& shape, index2_t const& i) -> index_t
{
    return i.y * shape.x + i.x;
}

inline constexpr auto unravel(index2_t const& shape, index_t const& i) -> index2_t
{
    return { i % shape.x, i / shape.x };
}

inline constexpr auto stride(index2_t const& shape) -> index_t
{
    return shape.x;
}
