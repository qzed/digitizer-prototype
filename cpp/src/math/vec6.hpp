#pragma once

#include "../math.hpp"
#include "../utils/access.hpp"


namespace math {

template<class T>
struct vec6_t {
    std::array<T, 6> data;

    constexpr auto operator[] (index_t i) -> T&;
    constexpr auto operator[] (index_t i) const -> T const&;
};


template<class T>
inline constexpr auto vec6_t<T>::operator[] (index_t i) -> T&
{
    return utils::access::access(data, i);
}

template<class T>
inline constexpr auto vec6_t<T>::operator[] (index_t i) const -> T const&
{
    return utils::access::access(data, i);
}

} /* namespace math */
