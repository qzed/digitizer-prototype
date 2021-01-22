#pragma once

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


template<class T, class I>
inline constexpr auto checked(T& collection, I i) -> typename T::value_type&
{
    return collection.at(i);
}

template<class T, class I>
inline constexpr auto checked(T const& collection, I i) -> typename T::value_type const&
{
    return collection.at(i);
}


template<class T, class I>
inline constexpr auto unchecked(T& collection, I i) -> typename T::value_type&
{
    return collection[i];
}

template<class T, class I>
inline constexpr auto unchecked(T const& collection, I i) -> typename T::value_type const&
{
    return collection[i];
}


template<class T, class I>
inline constexpr auto access(T& collection, I i) -> typename T::value_type&
{
    if constexpr (mode == access_mode::checked) {
        return checked(collection, i);
    } else {
        return unchecked(collection, i);
    }
}

template<class T, class I>
inline constexpr auto access(T const& collection, I i) -> typename T::value_type const&
{
    if constexpr (mode == access_mode::checked) {
        return checked(collection, i);
    } else {
        return unchecked(collection, i);
    }
}


template<class T, class I>
inline constexpr auto access(T& collection, I i, bool cond, char const* msg) -> typename T::value_type&
{
    if constexpr (mode == access_mode::checked) {
        if (!cond) {
            throw std::out_of_range { msg };
        }

        return checked(collection, i);

    } else {
        return unchecked(collection, i);
    }
}

template<class T, class I>
inline constexpr auto access(T const& collection, I i, bool cond, char const* msg) -> typename T::value_type const&
{
    if constexpr (mode == access_mode::checked) {
        if (!cond) {
            throw std::out_of_range { msg };
        }

        return checked(collection, i);

    } else {
        return unchecked(collection, i);
    }
}

} /* namespace utils::access */
