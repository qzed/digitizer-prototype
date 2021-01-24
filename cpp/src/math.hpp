#pragma once

#include "types.hpp"

#include "math/mat2.hpp"
#include "math/num.hpp"
#include "math/poly2.hpp"
#include "math/vec2.hpp"

#include "utils/access.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <numeric>


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


template<typename T, index_t Nx, index_t Ny>
struct kernel {
    std::array<T, Nx * Ny> data;

    using Scalar = T;

    constexpr auto operator[] (index2_t const& i) const noexcept -> T const&;
    constexpr auto operator[] (index2_t const& i) noexcept -> T&;

    constexpr auto operator[] (index_t const& i) const noexcept -> T const&;
    constexpr auto operator[] (index_t const& i) noexcept -> T&;

    auto begin() noexcept -> T*;
    auto end() noexcept -> T*;

    auto begin() const noexcept -> T const*;
    auto end() const noexcept -> T const*;

    auto cbegin() const noexcept -> T const*;
    auto cend() const noexcept -> T const*;

    constexpr auto shape() const noexcept -> index2_t;
};

template<typename T, index_t Nx, index_t Ny>
auto operator<< (std::ostream& os, kernel<T, Nx, Ny> const& k) -> std::ostream&
{
    os << "[[" << k[{0, 0}];

    for (index_t x = 1; x < Nx; ++x) {
        os << ", " << k[{x, 0}];
    }

    for (index_t y = 1; y < Ny; ++y) {
        os << "], [" << k[{0, y}];

        for (index_t x = 1; x < Nx; ++x) {
            os << ", " << k[{x, y}];
        }
    }

    return os << "]]";
}

template<typename T, index_t Nx, index_t Ny>
constexpr auto kernel<T, Nx, Ny>::operator[] (index2_t const& i) const noexcept -> T const&
{
    return utils::access::access<T>(this->data, ravel, { Nx, Ny }, i);
}

template<typename T, index_t Nx, index_t Ny>
constexpr auto kernel<T, Nx, Ny>::operator[] (index2_t const& i) noexcept -> T&
{
    return utils::access::access<T>(this->data, ravel, { Nx, Ny }, i);
}

template<typename T, index_t Nx, index_t Ny>
constexpr auto kernel<T, Nx, Ny>::operator[] (index_t const& i) const noexcept -> T const&
{
    return utils::access::access<T>(this->data, Nx * Ny, i);
}

template<typename T, index_t Nx, index_t Ny>
constexpr auto kernel<T, Nx, Ny>::operator[] (index_t const& i) noexcept -> T&
{
    return utils::access::access<T>(this->data, Nx * Ny, i);
}

template<typename T, index_t Nx, index_t Ny>
auto kernel<T, Nx, Ny>::begin() noexcept -> T*
{
    return &this->data[0];
}

template<typename T, index_t Nx, index_t Ny>
auto kernel<T, Nx, Ny>::end() noexcept -> T*
{
    return &this->data[Nx * Ny];
}

template<typename T, index_t Nx, index_t Ny>
auto kernel<T, Nx, Ny>::begin() const noexcept -> T const*
{
    return &this->data[0];
}

template<typename T, index_t Nx, index_t Ny>
auto kernel<T, Nx, Ny>::end() const noexcept -> T const*
{
    return &this->data[Nx * Ny];
}

template<typename T, index_t Nx, index_t Ny>
auto kernel<T, Nx, Ny>::cbegin() const noexcept -> T const*
{
    return &this->data[0];
}

template<typename T, index_t Nx, index_t Ny>
auto kernel<T, Nx, Ny>::cend() const noexcept -> T const*
{
    return &this->data[Nx * Ny];
}

template<typename T, index_t Nx, index_t Ny>
constexpr auto kernel<T, Nx, Ny>::shape() const noexcept -> index2_t
{
    return { Nx, Ny };
}
