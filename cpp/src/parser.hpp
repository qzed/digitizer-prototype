#pragma once

#include "types.hpp"

#include <vector>
#include <string>


struct ipts_data {
    u32 type;
    u32 size;
    u32 buffer;
    u8 reserved[52];
} __attribute__ ((packed));

struct ipts_payload {
    u32 counter;
    u32 frames;
    u8 reserved[4];
} __attribute__ ((packed));

struct ipts_payload_frame {
    u16 index;
    u16 type;
    u32 size;
    u8 reserved[8];
} __attribute__ ((packed));

struct ipts_report {
    u16 type;
    u16 size;
} __attribute__ ((packed));

struct ipts_stylus_report_s {
    u8 elements;
    u8 reserved[3];
    u32 serial;
} __attribute__ ((packed));

struct ipts_stylus_report_n {
    u8 elements;
    u8 reserved[3];
} __attribute__ ((packed));

struct ipts_stylus_data_v1 {
    u8 reserved1[4];
    u8 mode;
    u16 x;
    u16 y;
    u16 pressure;
    u8 reserved2;
} __attribute__ ((packed));

struct ipts_stylus_data_v2 {
    u16 timestamp;
    u16 mode;
    u16 x;
    u16 y;
    u16 pressure;
    u16 altitude;
    u16 azimuth;
    u8 reserved[2];
} __attribute__ ((packed));

struct ipts_heatmap_dim {
    u8 height;
    u8 width;
    u8 y_min;
    u8 y_max;
    u8 x_min;
    u8 x_max;
    u8 z_min;
    u8 z_max;
} __attribute__ ((packed));


struct slice_index {
    std::size_t begin;
    std::size_t end;
};

template<typename T>
struct slice {
    T const* begin;
    T const* end;

    auto size() const noexcept -> std::size_t;
    auto operator[] (slice_index i) const noexcept -> slice<T>;
};

template<typename T>
auto slice<T>::size() const noexcept -> std::size_t
{
    return end - begin;
}

template<typename T>
auto slice<T>::operator[] (slice_index i) const noexcept -> slice<T>
{
    return { this->begin + i.begin, std::min(this->begin + i.end, this->end) };
}


template<typename T>
auto make_slice(std::vector<T> const& v) -> slice<T>
{
    return { v.data(), v.data() + v.size() };
}


class parser_exception : public std::exception {
private:
    std::string m_reason;

public:
    parser_exception(std::string reason);

    auto what() const noexcept -> const char*;
};

parser_exception::parser_exception(std::string reason)
    : m_reason{reason}
{}

auto parser_exception::what() const noexcept -> const char*
{
    return m_reason.c_str();
}


class parser_base {
public:
    virtual ~parser_base() = default;

protected:
    void do_parse(slice<u8> data);

    auto parse_data(slice<u8> data) -> slice<u8>;
    void parse_data_payload(ipts_data const& header, slice<u8> data);
    auto parse_payload_frame(ipts_payload const& header, slice<u8> data) -> slice<u8>;
    void parse_payload_frame_reports(ipts_payload_frame const& header, slice<u8> data);
    auto parse_report(ipts_payload_frame const& header, slice<u8> data) -> slice<u8>;
    void parse_report_heatmap_dim(ipts_report const& header, slice<u8> data);
    void parse_report_heatmap(ipts_report const& header, slice<u8> data);

    virtual void on_heatmap_dim(ipts_heatmap_dim const& dim);
    virtual void on_heatmap(slice<u8> const& data);
};


void parser_base::do_parse(slice<u8> data)
{
    while (data.size()) {
        data = parse_data(data);
    }
}

auto parser_base::parse_data(slice<u8> data) -> slice<u8>
{
    ipts_data hdr;
    slice<u8> pld;

    std::copy(data.begin, data.begin + sizeof(hdr), reinterpret_cast<u8*>(&hdr));
    pld = { data.begin + sizeof(hdr), data.begin + sizeof(hdr) + hdr.size };

    if (data.begin + sizeof(hdr) + hdr.size > data.end)
        throw parser_exception{"EOF"};

    switch (hdr.type) {
    case 0x00:
        parse_data_payload(hdr, pld);
        break;

    default:
        break;
    };

    return { pld.end, data.end };
}

void parser_base::parse_data_payload(ipts_data const& header, slice<u8> data)
{
    ipts_payload hdr;
    slice<u8> pld;

    std::copy(data.begin, data.begin + sizeof(hdr), reinterpret_cast<u8*>(&hdr));
    pld = { data.begin + sizeof(hdr), data.end };

    for (unsigned int i = 0; i < hdr.frames; ++i) {
        pld = parse_payload_frame(hdr, pld);
    }
}

auto parser_base::parse_payload_frame(ipts_payload const& header, slice<u8> data) -> slice<u8>
{
    ipts_payload_frame hdr;
    slice<u8> pld;

    std::copy(data.begin, data.begin + sizeof(hdr), reinterpret_cast<u8*>(&hdr));
    pld = { data.begin + sizeof(hdr), data.begin + sizeof(hdr) + hdr.size };

    if (data.begin + sizeof(hdr) + hdr.size > data.end)
        throw parser_exception{"EOF"};

    switch (hdr.type)
    {
    case 0x06:
    case 0x07:
    case 0x08:
        parse_payload_frame_reports(hdr, pld);
        break;

    default:
        break;
    }

    return { pld.end, data.end };
}

void parser_base::parse_payload_frame_reports(ipts_payload_frame const& header, slice<u8> data)
{
    while (data.size() >= sizeof(ipts_report)) {
        data = parse_report(header, data);
    }
}

auto parser_base::parse_report(ipts_payload_frame const& header, slice<u8> data) -> slice<u8>
{
    ipts_report hdr;
    slice<u8> pld;

    std::copy(data.begin, data.begin + sizeof(hdr), reinterpret_cast<u8*>(&hdr));
    pld = { data.begin + sizeof(hdr), data.begin + sizeof(hdr) + hdr.size };

    switch (hdr.type)
    {
    case 0x403:
        parse_report_heatmap_dim(hdr, pld);
        break;

    case 0x425:
        parse_report_heatmap(hdr, pld);
        break;

    default:
        break;
    }

    return { pld.end, data.end };
}

void parser_base::parse_report_heatmap_dim(ipts_report const& header, slice<u8> data)
{
    ipts_heatmap_dim dim;

    std::copy(data.begin, data.begin + sizeof(dim), reinterpret_cast<u8*>(&dim));

    on_heatmap_dim(dim);
}

void parser_base::parse_report_heatmap(ipts_report const& header, slice<u8> data)
{
    on_heatmap(data);
}


void parser_base::on_heatmap_dim(ipts_heatmap_dim const& dim)
{}

void parser_base::on_heatmap(slice<u8> const& dim)
{}
