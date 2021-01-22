#pragma once

#include "../math.hpp"
#include "../utils/access.hpp"


namespace math {

template<class T>
struct vec6 {
    std::array<T, 6> data;

    constexpr auto operator[] (index i) -> T&;
    constexpr auto operator[] (index i) const -> T const&;
};


template<typename T>
inline constexpr auto vec6<T>::operator[] (index i) -> T&
{
    return utils::access::access(data, i);
}

template<typename T>
inline constexpr auto vec6<T>::operator[] (index i) const -> T const&
{
    return utils::access::access(data, i);
}

} /* namespace math */
