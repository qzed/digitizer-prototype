#pragma once

#include "types.hpp"

#include <sstream>
#include <stdexcept>


namespace utils::access {

enum class access_mode {
    checked,
    unchecked,
};


#ifdef IPTSD_CONFIG_ACCESS_CHECKS
inline static constexpr access_mode mode = access_mode::checked;
#else
inline static constexpr access_mode mode = access_mode::unchecked;
#endif


inline void ensure(index_t size, index_t i)
{
    if constexpr (mode == access_mode::unchecked) {
        return;
    }

    if (0 <= i && i < size) {
        return;
    }

    auto s = std::ostringstream{};
    s << "invalid access: size is " << size << ", index is " << i;

    throw std::out_of_range { s.str() };
}

inline void ensure(index2_t shape, index2_t i)
{
    if constexpr (mode == access_mode::unchecked) {
        return;
    }

    if (0 <= i.x && i.x < shape.x && 0 <= i.y && i.y < shape.y) {
        return;
    }

    auto s = std::ostringstream{};
    s << "invalid access: size is " << shape << ", index is " << i;

    throw std::out_of_range { s.str() };
}


template<class V, class T>
inline constexpr auto access(T const& data, index_t size, index_t i) -> V const&
{
    ensure(size, i);

    return data[i];
}

template<class V, class T>
inline constexpr auto access(T& data, index_t size, index_t i) -> V&
{
    ensure(size, i);

    return data[i];
}

template<class V, class T, class F>
inline constexpr auto access(T const& data, F ravel, index2_t shape, index2_t i) -> V const&
{
    ensure(shape, i);

    return data[ravel(shape, i)];
}

template<class V, class T, class F>
inline constexpr auto access(T& data, F ravel, index2_t shape, index2_t i) -> V&
{
    ensure(shape, i);

    return data[ravel(shape, i)];
}

} /* namespace utils::access */
