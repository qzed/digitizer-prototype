#pragma once

#include "math.hpp"
#include "cmap.hpp"

#include <exception>
#include <utility>
#include <cairo/cairo.h>


namespace cairo {

using status_t = cairo_status_t;

class surface;
class pattern;
class matrix;


class exception : public std::exception {
private:
    status_t m_code;

public:
    exception(status_t code);

    auto what() const noexcept -> const char*;
    auto code() const noexcept -> status_t;
};

exception::exception(status_t code)
    : m_code{code}
{}

auto exception::what() const noexcept -> const char*
{
    return cairo_status_to_string(m_code);
}

auto exception::code() const noexcept -> status_t
{
    return m_code;
}


enum class format {
    argb32   = CAIRO_FORMAT_ARGB32,
    rgba128f = CAIRO_FORMAT_RGBA128F,
};

enum class filter {
    nearest = CAIRO_FILTER_NEAREST,
};


class cairo {
private:
    cairo_t* m_raw;

public:
    cairo();
    cairo(cairo_t* raw);
    cairo(cairo const& other);
    cairo(cairo&& other);
    ~cairo();

    static auto create(surface& surface) -> cairo;

    void operator=(cairo const& rhs);
    void operator=(cairo&& rhs);

    auto raw() -> cairo_t*;
    auto operator* () -> cairo_t*;

    auto status() const -> status_t;

    void set_source(pattern& p);
    void set_source(cmap::srgb rgb);
    void set_source(cmap::srgba rgba);
    void set_source(surface& src, vec2<f64> origin);
    auto get_source() -> pattern;

    void set_source_filter(filter f);

    void paint();
    void fill();

    void save();
    void restore();

    void scale(vec2<f64> s);

    void move_to(vec2<f64> pos);
    void line_to(vec2<f64> pos);
    void rectangle(vec2<f64> origin, vec2<f64> size);
};


class surface {
private:
    cairo_surface_t* m_raw;

public:
    surface();
    surface(cairo_surface_t* raw);
    surface(surface const& other);
    surface(surface&& other);
    ~surface();

    void operator=(surface const& rhs);
    void operator=(surface&& rhs);

    auto raw() -> cairo_surface_t*;
    auto operator* () -> cairo_surface_t*;

    auto status() const -> status_t;
};


class pattern {
private:
    cairo_pattern_t* m_raw;

public:
    pattern();
    pattern(cairo_pattern_t* raw);
    pattern(pattern const& other);
    pattern(pattern&& other);
    ~pattern();

    static auto create_for_surface(surface &surface) -> pattern;

    void operator=(pattern const& rhs);
    void operator=(pattern&& rhs);

    auto raw() -> cairo_pattern_t*;
    auto operator* () -> cairo_pattern_t*;

    auto status() const -> status_t;

    void set_matrix(matrix &m);
    void set_filter(filter f);
};


class matrix {
private:
    cairo_matrix_t m_raw;

public:
    matrix();
    matrix(cairo_matrix_t m);

    static auto identity() -> matrix;

    auto raw() -> cairo_matrix_t*;
    auto operator* () -> cairo_matrix_t*;

    void translate(vec2<f64> v);
    void scale(vec2<f64> v);
};


cairo::cairo()
    : m_raw{nullptr}
{}

cairo::cairo(cairo_t* raw)
    : m_raw{raw}
{}

cairo::cairo(cairo const& other)
    : m_raw{other.m_raw ? cairo_reference(other.m_raw) : nullptr}
{}

cairo::cairo(cairo&& other)
    : m_raw{std::exchange(other.m_raw, nullptr)}
{}

cairo::~cairo()
{
    if (m_raw) {
        cairo_destroy(m_raw);
    }
}

auto cairo::create(surface& target) -> cairo
{
    cairo cr { cairo_create(target.raw()) };

    if (cr.status() != CAIRO_STATUS_SUCCESS) {
        throw exception{cr.status()};
    }

    return cr;
}

void cairo::operator=(cairo const& rhs)
{
    m_raw = rhs.m_raw ? cairo_reference(rhs.m_raw) : nullptr;
}

void cairo::operator=(cairo&& rhs)
{
    m_raw = std::exchange(rhs.m_raw, nullptr);
}

auto cairo::raw() -> cairo_t*
{
    return m_raw;
}

auto cairo::operator* () -> cairo_t*
{
    return m_raw;
}

auto cairo::status() const -> cairo_status_t
{
    return cairo_status(m_raw);
}

void cairo::set_source(pattern& p)
{
    cairo_set_source(m_raw, *p);
}

void cairo::set_source(cmap::srgb c)
{
    cairo_set_source_rgb(m_raw, c.r, c.g, c.b);
}

void cairo::set_source(cmap::srgba c)
{
    cairo_set_source_rgba(m_raw, c.r, c.g, c.b, c.a);
}

void cairo::set_source(surface& src, vec2<f64> origin)
{
    cairo_set_source_surface(m_raw, *src, origin.x, origin.y);
}

auto cairo::get_source() -> pattern
{
    return { cairo_pattern_reference(cairo_get_source(m_raw)) };
}

void cairo::set_source_filter(filter f)
{
    cairo_pattern_set_filter(cairo_get_source(m_raw), static_cast<cairo_filter_t>(f));
}

void cairo::paint()
{
    cairo_paint(m_raw);
}

void cairo::fill()
{
    cairo_fill(m_raw);
}

void cairo::save()
{
    cairo_save(m_raw);
}

void cairo::restore()
{
    cairo_restore(m_raw);
}

void cairo::scale(vec2<f64> s)
{
    cairo_scale(m_raw, s.x, s.y);
}

void cairo::move_to(vec2<f64> pos)
{
    cairo_move_to(m_raw, pos.x, pos.y);
}

void cairo::line_to(vec2<f64> pos)
{
    cairo_line_to(m_raw, pos.x, pos.y);
}

void cairo::rectangle(vec2<f64> origin, vec2<f64> size)
{
    cairo_rectangle(m_raw, origin.x, origin.y, size.x, size.y);
}


surface::surface()
    : m_raw{nullptr}
{}

surface::surface(cairo_surface_t* raw)
    : m_raw{raw}
{}

surface::surface(surface const& other)
    : m_raw{other.m_raw ? cairo_surface_reference(other.m_raw) : nullptr}
{}

surface::surface(surface&& other)
    : m_raw{std::exchange(other.m_raw, nullptr)}
{}

surface::~surface()
{
    if (m_raw) {
        cairo_surface_destroy(m_raw);
    }
}

void surface::operator=(surface const& rhs)
{
    m_raw = rhs.m_raw ? cairo_surface_reference(rhs.m_raw) : nullptr;
}

void surface::operator=(surface&& rhs)
{
    m_raw = std::exchange(rhs.m_raw, nullptr);
}

auto surface::raw() -> cairo_surface_t*
{
    return m_raw;
}

auto surface::operator* () -> cairo_surface_t*
{
    return m_raw;
}

auto surface::status() const -> cairo_status_t
{
    return cairo_surface_status(m_raw);
}


pattern::pattern()
    : m_raw{}
{}

pattern::pattern(cairo_pattern_t* raw)
    : m_raw{raw}
{}

pattern::pattern(pattern const& other)
    : m_raw{other.m_raw ? cairo_pattern_reference(other.m_raw) : nullptr}
{}

pattern::pattern(pattern&& other)
    : m_raw{std::exchange(other.m_raw, nullptr)}
{}

pattern::~pattern()
{
    if (m_raw) {
        cairo_pattern_destroy(m_raw);
    }
}

auto pattern::create_for_surface(surface &surface) -> pattern
{
    return { cairo_pattern_create_for_surface(*surface) };
}

void pattern::operator=(pattern const& rhs)
{
    m_raw = rhs.m_raw ? cairo_pattern_reference(rhs.m_raw) : nullptr;
}

void pattern::operator=(pattern&& rhs)
{
    m_raw = std::exchange(rhs.m_raw, nullptr);
}

auto pattern::raw() -> cairo_pattern_t*
{
    return m_raw;
}

auto pattern::operator* () -> cairo_pattern_t*
{
    return m_raw;
}

auto pattern::status() const -> status_t
{
    return cairo_pattern_status(m_raw);
}

void pattern::set_matrix(matrix &m)
{
    return cairo_pattern_set_matrix(m_raw, *m);
}

void pattern::set_filter(filter f)
{
    return cairo_pattern_set_filter(m_raw, static_cast<cairo_filter_t>(f));
}


matrix::matrix()
    : m_raw{}
{}

matrix::matrix(cairo_matrix_t m)
    : m_raw{m}
{}

auto matrix::identity() -> matrix
{
    auto m = matrix{};
    cairo_matrix_init_identity(&m.m_raw);
    return m;
}

auto matrix::raw() -> cairo_matrix_t*
{
    return &m_raw;
}

auto matrix::operator* () -> cairo_matrix_t*
{
    return &m_raw;
}

void matrix::translate(vec2<f64> v)
{
    cairo_matrix_translate(&m_raw, v.x, v.y);
}

void matrix::scale(vec2<f64> s)
{
    cairo_matrix_scale(&m_raw, s.x, s.y);
}


template<typename T>
constexpr auto pixel_format() -> format;

template<>
constexpr auto pixel_format<cmap::srgba>() -> format
{
    return format::rgba128f;
}


auto image_surface_create(format fmt, vec2<i32> shape) -> surface
{
    surface s { cairo_image_surface_create(static_cast<cairo_format_t>(fmt), shape.x, shape.y) };

    if (s.status() != CAIRO_STATUS_SUCCESS) {
        throw exception{s.status()};
    }

    return s;
}

auto format_stride_for_width(format fmt, int width) -> int
{
    int stride = cairo_format_stride_for_width(static_cast<cairo_format_t>(fmt), width);

    if (stride < 0) {
        throw exception{CAIRO_STATUS_INVALID_STRIDE};
    }

    return stride;
}

template<typename T>
auto image_surface_create(image<T>& image) -> surface
{
    auto const format = pixel_format<T>();
    auto const shape = image.shape();
    auto const data = reinterpret_cast<u8*>(image.data());

    auto const stride = format_stride_for_width(format, shape.x);
    auto const cf = static_cast<cairo_format_t>(format);
    auto const ptr = cairo_image_surface_create_for_data(data, cf, shape.x, shape.y, stride);
    auto const s = surface(ptr);

    if (s.status() != CAIRO_STATUS_SUCCESS) {
        throw exception{s.status()};
    }

    return s;
}

} /* namespace cairo */
