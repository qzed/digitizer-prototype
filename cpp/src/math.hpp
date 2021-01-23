#pragma once

#include "types.hpp"
#include "utils/access.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <numeric>


template<typename T>
auto zero() -> T;

template<>
auto zero() -> f32
{
    return 0.0;
}

template<>
auto zero() -> f64
{
    return 0.0f;
}


template<typename T>
struct vec2 {
    T x, y;

    using Scalar = T;
};

template<typename T>
auto operator<< (std::ostream& os, vec2<T> const& v) -> std::ostream&
{
    return os << "[" << v.x << ", " << v.y << "]";
}

template<typename U, typename V>
auto operator== (vec2<U> const& a, vec2<V> const& b) noexcept -> bool
{
    return a.x == b.x && a.y == b.y;
}

template<typename U, typename V>
auto operator!= (vec2<U> const& a, vec2<V> const& b) noexcept -> bool
{
    return !(a == b);
}

template<typename U, typename V>
auto operator+ (vec2<U> const& a, vec2<V> const& b) noexcept -> vec2<decltype(a.x + b.x)>
{
    return { a.x + b.x, a.y + b.y };
}

template<typename U, typename V>
auto operator- (vec2<U> const& a, vec2<V> const& b) noexcept -> vec2<decltype(a.x - b.x)>
{
    return { a.x - b.x, a.y - b.y };
}

template<typename U, typename V>
auto mul (vec2<U> const& a, vec2<V> const& b) noexcept -> vec2<decltype(a.x * b.x)>
{
    return { a.x * b.x, a.y * b.y };
}

template<typename U>
auto sum (vec2<U> const& a) noexcept
{
    return a.x + a.y;
}

template<typename U>
auto prod (vec2<U> const& a) noexcept
{
    return a.x * a.y;
}

template<typename U>
auto l2norm (vec2<U> const& a) noexcept -> U
{
    return std::sqrt(a.x * a.x + a.y * a.y);
}

template<typename U, typename V>
auto dot (vec2<U> const& a, vec2<V> const& b) noexcept
{
    return sum(mul(a, b));
}


template<>
auto zero() -> vec2<f32>
{
    return { 0.0f, 0.0f };
}

template<>
auto zero() -> vec2<f64>
{
    return { 0.0, 0.0 };
}


template<typename T>
struct mat2s {
    T xx, xy, yy;

    using Scalar = T;

    static constexpr auto identity() -> mat2s<T>;

    constexpr auto operator+= (mat2s<T> const& rhs) noexcept -> mat2s<T>&;
};

template<typename T>
constexpr auto mat2s<T>::identity() -> mat2s<T>
{
    return { static_cast<T>(1), static_cast<T>(0), static_cast<T>(1) };
}

template<typename T>
constexpr auto mat2s<T>::operator+= (mat2s<T> const& rhs) noexcept -> mat2s<T>&
{
    this->xx += rhs.xx;
    this->xy += rhs.xy;
    this->yy += rhs.yy;
    return *this;
}

template<typename T>
auto operator<< (std::ostream& os, mat2s<T> const& m) -> std::ostream&
{
    return os << "[[" << m.xx << ", " << m.xy << "], [" << m.xy << ", " << m.yy << "]]";
}

template<typename U, typename V>
constexpr auto operator+ (mat2s<U> const& a, mat2s<V> const& b) noexcept -> mat2s<decltype(a.xx + b.xx)>
{
    return { a.xx + b.xx, a.xy + b.xy, a.yy + b.yy };
}

template<typename T>
constexpr auto operator* (mat2s<T> const& a, T s) noexcept -> mat2s<T>
{
    return { a.xx * s, a.xy * s, a.yy * s };
}

template<typename U, typename V>
constexpr auto mul(mat2s<U> const& a, mat2s<V> const& b) noexcept -> mat2s<decltype(a.xx * b.xx)>
{
    return { a.xx * b.xx, a.xy * b.xy, a.yy * b.yy };
}


template<typename T>
constexpr auto xtmx(mat2s<T> const& m, vec2<T> const& v) noexcept -> T
{
    return v.x * v.x * m.xx + static_cast<T>(2) * v.x * v.y * m.xy + v.y * v.y * m.yy;
}



template<>
auto zero() -> mat2s<f32>
{
    return { 0.0f, 0.0f, 0.0f };
}

template<>
auto zero() -> mat2s<f64>
{
    return { 0.0, 0.0, 0.0 };
}


using index_t = i32;
using index2_t = vec2<index_t>;

constexpr auto ravel(index2_t const& shape, index2_t const& i) noexcept -> index_t
{
    return i.y * shape.x + i.x;
}

constexpr auto unravel(index2_t const& shape, index_t const& i) noexcept -> index2_t
{
    return { i % shape.x, i / shape.x };
}

constexpr auto stride(index2_t const& shape) noexcept -> index_t
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
    return std::accumulate(obj.begin(), obj.end(), static_cast<typename T::Scalar>(0));
}

template<typename T>
auto average(T const& obj) -> typename T::Scalar
{
    return sum(obj) / static_cast<typename T::Scalar>(prod(obj.shape()));
}

template<typename T>
void sub0(T& obj, typename T::Scalar s)
{
    transform_inplace(obj, [&](auto const& x) {
        return std::max(x - s, static_cast<typename T::Scalar>(0));
    });
}


template<typename T>
auto trace(mat2s<T> const& m) -> T
{
    return m.xx + m.yy;
}

template<typename T>
auto det(mat2s<T> const& m) -> T
{
    return m.xx * m.yy - m.xy * m.xy;
}

template<typename T>
auto inv(mat2s<T> const& m, T eps=zero<T>()) -> std::optional<mat2s<T>>
{
    auto const d = det(m);

    if (std::abs(d) <= eps)
        return std::nullopt;

    return {{ m.yy / d, -m.xy / d, m.xx / d }};
}


template<typename T>
auto solve_quadratic(T a, T b, T c, T eps=static_cast<T>(1e-20)) -> std::array<T, 2>
{
    if (std::abs(a) <= eps) {           // case: bx + c = 0
        return { -c / b, zero<T>() };
    }

    if (std::abs(c) <= eps) {           // case: ax^2 + bx = 0
        return { -b / a, zero<T>() };
    }

    // Note: Does not prevent potential overflows in b^2

    // stable(-ish) algorithm: prevent cancellation
    auto const r1 = (-b - std::copysign(std::sqrt(b * b - 4 * a * c), b)) / (2 * a);
    auto const r2 = c / (a * r1);

    return { r1, r2 };
}


template<typename T>
struct eigen {
    std::array<T, 2>       w;
    std::array<vec2<T>, 2> v;
};

template<typename M, typename S = typename M::Scalar>
auto eigenvalues(M const& m, S eps=static_cast<S>(1e-20)) -> std::array<S, 2>
{
    return solve_quadratic<S>(1, -trace(m), det(m), eps);
}

template<class T>
auto eigenvector(mat2s<T> const& m, T ew) -> vec2<T>
{
    auto ev = vec2<T>{};

    /*
     * This 'if' should prevent two problems:
     * 1. Cancellation due to small values in subtraction.
     * 2. The vector being { 0, 0 }.
     */
    if (std::abs(m.xx - ew) > std::abs(m.yy - ew)) {
        ev = { -m.xy, m.xx - ew };
    } else {
        ev = { m.yy - ew, -m.xy };
    }

    auto const n = l2norm(ev);
    return { ev.x / n, ev.y / n };
}


template<typename M, typename S = typename M::Scalar>
auto eigenvectors(M const& m, S eps=static_cast<S>(1e-20)) -> eigen<S>
{
    auto const [ew1, ew2] = eigenvalues(m, eps);

    auto ev1 = eigenvector(m, ew1);
    auto ev2 = eigenvector(m, ew2);

    return {
        { ew1, ew2 },
        { ev1, ev2 },
    };
}
