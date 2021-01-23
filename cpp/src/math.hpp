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


using math::vec2_t;
using math::mat2s_t;


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
    return utils::access::access(this->data, ravel({Nx, Ny}, i));
}

template<typename T, index_t Nx, index_t Ny>
constexpr auto kernel<T, Nx, Ny>::operator[] (index2_t const& i) noexcept -> T&
{
    return utils::access::access(this->data, ravel({Nx, Ny}, i));
}

template<typename T, index_t Nx, index_t Ny>
constexpr auto kernel<T, Nx, Ny>::operator[] (index_t const& i) const noexcept -> T const&
{
    return utils::access::access(this->data, i);
}

template<typename T, index_t Nx, index_t Ny>
constexpr auto kernel<T, Nx, Ny>::operator[] (index_t const& i) noexcept -> T&
{
    return utils::access::access(this->data, i);
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


template<typename T>
class image {
private:
    index2_t m_shape;
    std::vector<T> m_data;

public:
    using Scalar = T;

    image(index2_t const& shape);

    image(image<T> const& other) noexcept = default;
    image(image<T>&& other) noexcept = default;

    auto operator= (image<T> const& rhs) noexcept -> image<T>& = default;
    auto operator= (image<T>&& rhs) noexcept -> image<T>& = default;

    auto operator[] (index2_t const& i) const noexcept -> T const&;
    auto operator[] (index2_t const& i) noexcept -> T&;

    auto operator[] (index_t const& i) const noexcept -> T const&;
    auto operator[] (index_t const& i) noexcept -> T&;

    auto shape() const noexcept -> index2_t const&;

    auto data() noexcept -> T*;
    auto data() const noexcept -> T const*;

    auto begin() noexcept -> auto;
    auto end() noexcept -> auto;

    auto begin() const noexcept -> auto const;
    auto end() const noexcept -> auto const;

    auto cbegin() const noexcept -> auto const;
    auto cend() const noexcept -> auto const;
};

template<typename T>
image<T>::image(index2_t const& shape)
    : m_shape { shape }
    , m_data(static_cast<std::size_t>(m_shape.x * m_shape.y))
{}

template<typename T>
auto image<T>::operator[] (index2_t const& i) const noexcept -> T const&
{
    return utils::access::access(m_data, ravel(m_shape, i));
}

template<typename T>
auto image<T>::operator[] (index2_t const& i) noexcept -> T&
{
    return utils::access::access(m_data, ravel(m_shape, i));
}

template<typename T>
auto image<T>::operator[] (index_t const& i) const noexcept -> T const&
{
    return utils::access::access(m_data, i);
}

template<typename T>
auto image<T>::operator[] (index_t const& i) noexcept -> T&
{
    return utils::access::access(m_data, i);
}

template<typename T>
auto image<T>::shape() const noexcept -> index2_t const&
{
    return m_shape;
}

template<typename T>
auto image<T>::data() noexcept -> T*
{
    return m_data.data();
}

template<typename T>
auto image<T>::data() const noexcept -> T const*
{
    return m_data.data();
}

template<typename T>
auto image<T>::cbegin() const noexcept -> auto const
{
    return m_data.cbegin();
}

template<typename T>
auto image<T>::begin() const noexcept -> auto const
{
    return m_data.cbegin();
}

template<typename T>
auto image<T>::begin() noexcept -> auto
{
    return m_data.begin();
}

template<typename T>
auto image<T>::cend() const noexcept -> auto const
{
    return m_data.cend();
}

template<typename T>
auto image<T>::end() const noexcept -> auto const
{
    return m_data.cend();
}

template<typename T>
auto image<T>::end() noexcept -> auto
{
    return m_data.end();
}


template<typename T>
auto minmax(image<T> const& img) -> std::pair<T, T>
{
    const auto [min, max] = std::minmax_element(img.begin(), img.end());

    return {*min, *max};
}


template<typename T, typename F>
void transform_inplace(T& obj, F op)
{
    std::transform(obj.begin(), obj.end(), obj.begin(), op);
}

template<typename T>
auto sum(T const& obj) -> typename T::Scalar
{
    return std::accumulate(obj.begin(), obj.end(), math::num<typename T::Scalar>::zero);
}

template<typename T>
auto average(T const& obj) -> typename T::Scalar
{
    return sum(obj) / static_cast<typename T::Scalar>(obj.shape().product());
}

template<typename T>
void sub0(T& obj, typename T::Scalar s)
{
    transform_inplace(obj, [&](auto const& x) {
        return std::max(x - s, math::num<typename T::Scalar>::zero);
    });
}
