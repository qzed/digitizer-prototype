#pragma once

#include <stdexcept>


// TODO: remove
// #define IPTSD_CONFIG_CHECK_ACCESS 1


namespace utils::access {

enum class access_mode {
    checked,
    unchecked,
};


#ifdef IPTSD_CONFIG_CHECK_ACCESS
inline static constexpr access_mode mode = access_mode::checked;
#else
inline static constexpr access_mode mode = access_mode::unchecked;
#endif


template<class T, class I>
inline constexpr auto checked(T& collection, I index) -> typename T::value_type&
{
    return collection.at(index);
}

template<class T, class I>
inline constexpr auto checked(T const& collection, I index) -> typename T::value_type const&
{
    return collection.at(index);
}


template<class T, class I>
inline constexpr auto unchecked(T& collection, I index) -> typename T::value_type&
{
    return collection[index];
}

template<class T, class I>
inline constexpr auto unchecked(T const& collection, I index) -> typename T::value_type const&
{
    return collection[index];
}


template<class T, class I>
inline constexpr auto access(T& collection, I index) -> typename T::value_type&
{
    if constexpr (mode == access_mode::checked) {
        return checked(collection, index);
    } else {
        return unchecked(collection, index);
    }
}

template<class T, class I>
inline constexpr auto access(T const& collection, I index) -> typename T::value_type const&
{
    if constexpr (mode == access_mode::checked) {
        return checked(collection, index);
    } else {
        return unchecked(collection, index);
    }
}


template<class T, class I>
inline constexpr auto access(T& collection, I index, bool cond, char const* msg) -> typename T::value_type&
{
    if constexpr (mode == access_mode::checked) {
        if (!cond) {
            throw std::out_of_range { msg };
        }

        return checked(collection, index);

    } else {
        return unchecked(collection, index);
    }
}

template<class T, class I>
inline constexpr auto access(T const& collection, I index, bool cond, char const* msg) -> typename T::value_type const&
{
    if constexpr (mode == access_mode::checked) {
        if (!cond) {
            throw std::out_of_range { msg };
        }

        return checked(collection, index);

    } else {
        return unchecked(collection, index);
    }
}

} /* namespace utils::access */
