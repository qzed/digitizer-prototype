#pragma once

#include "num.hpp"

#include "../math.hpp"
#include "../utils/access.hpp"


namespace math {

template<class T>
struct mat6_t {
    std::array<T, 6 * 6> data;

    constexpr static auto identity() -> mat6_t<T>;

    constexpr auto operator[] (index2_t i) -> T&;
    constexpr auto operator[] (index2_t i) const -> T const&;
};


template<class T>
inline constexpr auto mat6_t<T>::identity() -> mat6_t<T>
{
    auto const _0 = num<T>::zero;
    auto const _1 = num<T>::one;

    return {
        _1, _0, _0, _0, _0, _0,
        _0, _1, _0, _0, _0, _0,
        _0, _0, _1, _0, _0, _0,
        _0, _0, _0, _1, _0, _0,
        _0, _0, _0, _0, _1, _0,
        _0, _0, _0, _0, _0, _1,
    };
}

template<class T>
inline constexpr auto mat6_t<T>::operator[] (index2_t i) -> T&
{
    return utils::access::access(data, i.x * 6 + i.y,
                                 i.x >= 0 && i.x < 6 && i.y >= 0 && i.y < 6,
                                 "invalid matrix access");
}

template<class T>
inline constexpr auto mat6_t<T>::operator[] (index2_t i) const -> T const&
{
    return utils::access::access(data, i.x * 6 + i.y,
                                 i.x >= 0 && i.x < 6 && i.y >= 0 && i.y < 6,
                                 "invalid matrix access");
}

} /* namespace math */
