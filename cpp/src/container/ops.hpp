#pragma once

#include <algorithm>
#include <utility>


namespace container::ops {

template<class C>
inline auto minmax_element(C& container) -> std::pair<typename C::pointer, typename C::pointer>
{
    return std::minmax_element(container.begin(), container.end());
}

template<class C>
inline auto minmax_element(C const& container) -> std::pair<typename C::const_pointer, typename C::const_pointer>
{
    return std::minmax_element(container.begin(), container.end());
}

template<class C>
inline auto minmax(C const& container) -> std::pair<typename C::value_type, typename C::value_type>
{
    auto const [min, max] = minmax_element(container);
    return { *min, *max };
}


template<class S, class T, class F>
inline void transform(S const& source, T& target, F fn)
{
    assert(source.size() == target.size());

    std::transform(source.begin(), source.end(), target.begin(), fn);
}

template<class C, class F>
inline void transform(C& container, F fn)
{
    std::transform(container.begin(), container.end(), container.begin(), fn);
}

} /* namespace container::ops */
