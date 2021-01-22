#pragma once

#include "../math.hpp"
#include "../utils/access.hpp"


namespace math {

template<class T>
struct mat6 {
    std::array<T, 6 * 6> data;

    constexpr static auto identity() -> mat6<T>;

    constexpr auto operator[] (index2_t i) -> T&;
    constexpr auto operator[] (index2_t i) const -> T const&;
};


template<class T>
inline constexpr auto mat6<T>::identity() -> mat6<T>
{
    auto const _0 = static_cast<T>(0);
    auto const _1 = static_cast<T>(1);

    return {
        _1, _0, _0, _0, _0, _0,
        _0, _1, _0, _0, _0, _0,
        _0, _0, _1, _0, _0, _0,
        _0, _0, _0, _1, _0, _0,
        _0, _0, _0, _0, _1, _0,
        _0, _0, _0, _0, _0, _1,
    };
}

template<typename T>
inline constexpr auto mat6<T>::operator[] (index2_t i) -> T&
{
    return utils::access::access(data, i.x * 6 + i.y,
                                 i.x >= 0 && i.x < 6 && i.y >= 0 && i.y < 6,
                                 "invalid matrix access");
}

template<typename T>
inline constexpr auto mat6<T>::operator[] (index2_t i) const -> T const&
{
    return utils::access::access(data, i.x * 6 + i.y,
                                 i.x >= 0 && i.x < 6 && i.y >= 0 && i.y < 6,
                                 "invalid matrix access");
}

} /* namespace math */