#pragma once

#include <cstdint>
#include <iostream>


using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;


using index_t = int;

struct index2_t {
public:
    index_t x, y;

public:
    inline constexpr auto operator= (index2_t const& rhs) -> index2_t& = default;

    constexpr auto operator+= (index2_t const& v) -> index2_t&;
    constexpr auto operator-= (index2_t const& v) -> index2_t&;

    constexpr auto product() const -> index_t;
};


inline constexpr auto index2_t::operator+= (index2_t const& v) -> index2_t&
{
    this->x += v.x;
    this->y += v.y;
    return *this;
}

inline constexpr auto index2_t::operator-= (index2_t const& v) -> index2_t&
{
    this->x -= v.x;
    this->y -= v.y;
    return *this;
}


inline constexpr auto index2_t::product() const -> index_t
{
    return this->x * this->y;
}


inline auto operator<< (std::ostream& os, index2_t const& i) -> std::ostream&
{
    return os << "[" << i.x << ", " << i.y << "]";
}

inline constexpr auto operator== (index2_t const& a, index2_t const& b) -> bool
{
    return a.x == b.x && a.y == b.y;
}

inline constexpr auto operator!= (index2_t const& a, index2_t const& b) -> bool
{
    return !(a == b);
}

inline constexpr auto operator+ (index2_t const& a, index2_t const& b) -> index2_t
{
    return { a.x + b.x, a.y + b.y };
}

inline constexpr auto operator- (index2_t const& a, index2_t const& b) -> index2_t
{
    return { a.x - b.x, a.y - b.y };
}
