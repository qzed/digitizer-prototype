#pragma once

#include "types.hpp"
#include "utils/access.hpp"

#include <algorithm>
#include <utility>


namespace iptsd::container {

template<class T>
class Image {
public:
    using value_type             = T;
    using reference              = value_type&;
    using const_reference        = value_type const&;
    using pointer                = value_type*;
    using const_pointer          = value_type const*;
    using iterator               = T*;
    using const_iterator         = T const*;
    using reverse_iterator       = T*;
    using const_reverse_iterator = T const*;

public:
    Image();
    Image(index2_t size);
    Image(Image const& other);
    Image(Image&& other) noexcept;
    ~Image();

    auto operator= (Image<T> const& rhs) -> Image<T>&;
    auto operator= (Image<T>&& rhs) noexcept -> Image<T>&;

    auto size() const -> index2_t;
    auto stride() const -> index_t;

    auto data() -> pointer;
    auto data() const -> const_pointer;

    auto operator[] (index2_t const& i) const -> const_reference;
    auto operator[] (index2_t const& i) -> reference;

    auto operator[] (index_t const& i) const -> const_reference;
    auto operator[] (index_t const& i) -> reference;

    auto begin() -> iterator;
    auto end() -> iterator;

    auto begin() const -> const_iterator;
    auto end() const -> const_iterator;

    auto cbegin() const -> const_iterator;
    auto cend() const -> const_iterator;

    static constexpr auto ravel(index2_t size, index2_t i) -> index_t;
    static constexpr auto unravel(index2_t size, index_t i) -> index2_t;

private:
    index2_t m_size;
    T*       m_data;
};


template<class T>
Image<T>::Image()
    : m_size{0, 0}
    , m_data{nullptr}
{}

template<class T>
Image<T>::Image(index2_t size)
    : m_size{0, 0}
    , m_data{nullptr}
{
    m_data = new T[size.span()];
    m_size = size;
}

template<class T>
Image<T>::Image(Image const& other)
    : m_size{0, 0}
    , m_data{nullptr}
{
    // implement in terms of copy assignment operator to not leak any memory if copy throws...
    auto tmp = Image<T>{};
    tmp = other;

    std::swap(m_size, tmp.m_size);
    std::swap(m_data, tmp.m_data);
}

template<class T>
Image<T>::Image(Image&& other) noexcept
    : m_size{std::exchange(other.m_size, { 0, 0 })}
    , m_data{std::exchange(other.m_data, nullptr)}
{}

template<class T>
Image<T>::~Image()
{
    delete[] std::exchange(m_data, nullptr);
}

template<class T>
auto Image<T>::operator= (Image<T> const& rhs) -> Image<T>&
{
    if (m_size != rhs.m_size) {
        delete[] std::exchange(m_data, nullptr);

        m_data = new T[rhs.m_size.span()];
        m_size = rhs.m_size;
    }

    std::copy(rhs.begin(), rhs.end(), this->begin());

    return *this;
}

template<class T>
auto Image<T>::operator= (Image<T>&& rhs) noexcept -> Image<T>&
{
    delete[] m_data;

    m_data = std::exchange(rhs.m_data, nullptr);
    m_size = std::exchange(rhs.m_size, {0, 0});

    return *this;
}


template<class T>
inline auto Image<T>::size() const -> index2_t
{
    return m_size;
}

template<class T>
inline auto Image<T>::stride() const -> index_t
{
    return m_size.x;
}

template<class T>
inline auto Image<T>::data() -> pointer
{
    return m_data;
}

template<class T>
inline auto Image<T>::data() const -> const_pointer
{
    return m_data;
}

template<class T>
inline auto Image<T>::operator[] (index2_t const& i) const -> const_reference
{
    return utils::access::access<T>(m_data, ravel, m_size, i);
}

template<class T>
inline auto Image<T>::operator[] (index2_t const& i) -> reference
{
    return utils::access::access<T>(m_data, ravel, m_size, i);
}

template<class T>
inline auto Image<T>::operator[] (index_t const& i) const -> const_reference
{
    return utils::access::access<T>(m_data, m_size.span(), i);
}

template<class T>
inline auto Image<T>::operator[] (index_t const& i) -> reference
{
    return utils::access::access<T>(m_data, m_size.span(), i);
}

template<class T>
inline auto Image<T>::begin() -> iterator
{
    return &m_data[0];
}

template<class T>
inline auto Image<T>::end() -> iterator
{
    return &m_data[m_size.span()];
}

template<class T>
inline auto Image<T>::begin() const -> const_iterator
{
    return &m_data[0];
}

template<class T>
inline auto Image<T>::end() const -> const_iterator
{
    return &m_data[m_size.span()];
}

template<class T>
inline auto Image<T>::cbegin() const -> const_iterator
{
    return &m_data[0];
}

template<class T>
inline auto Image<T>::cend() const -> const_iterator
{
    return &m_data[m_size.span()];
}


template<class T>
inline constexpr auto Image<T>::ravel(index2_t size, index2_t i) -> index_t
{
    return i.y * size.x + i.x;
}

template<class T>
inline constexpr auto Image<T>::unravel(index2_t size, index_t i) -> index2_t
{
    return { i % size.x, i / size.x };
}

} /* namespace iptsd::container */


/* imports */
namespace iptsd {

using container::Image;

} /* namespace iptsd */
