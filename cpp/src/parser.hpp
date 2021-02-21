#pragma once

#include "types.hpp"

#include <vector>
#include <string>


namespace iptsd {

struct IptsData {
    u32 type;
    u32 size;
    u32 buffer;
    u8 reserved[52];
} __attribute__ ((packed));

struct IptsPayload {
    u32 counter;
    u32 frames;
    u8 reserved[4];
} __attribute__ ((packed));

struct IptsPayloadFrame {
    u16 index;
    u16 type;
    u32 size;
    u8 reserved[8];
} __attribute__ ((packed));

struct IptsReport {
    u16 type;
    u16 size;
} __attribute__ ((packed));

struct IptsTimestampReport {
    u16 unknown;
    u16 counter;
    u32 timestamp;
};

struct IptsStylusReportS {
    u8 elements;
    u8 reserved[3];
    u32 serial;
} __attribute__ ((packed));

struct IptsStylusReportN {
    u8 elements;
    u8 reserved[3];
} __attribute__ ((packed));

struct IptsStylusDataV1 {
    u8 reserved1[4];
    u8 mode;
    u16 x;
    u16 y;
    u16 pressure;
    u8 reserved2;
} __attribute__ ((packed));

struct IptsStylusDataV2 {
    u16 timestamp;
    u16 mode;
    u16 x;
    u16 y;
    u16 pressure;
    u16 altitude;
    u16 azimuth;
    u8 reserved[2];
} __attribute__ ((packed));

struct IptsHeatmapDim {
    u8 height;
    u8 width;
    u8 y_min;
    u8 y_max;
    u8 x_min;
    u8 x_max;
    u8 z_min;
    u8 z_max;
} __attribute__ ((packed));


struct SliceIndex {
    std::size_t begin;
    std::size_t end;
};

template<typename T>
struct Slice {
    T const* begin;
    T const* end;

    auto size() const noexcept -> std::size_t;
    auto operator[] (SliceIndex i) const noexcept -> Slice<T>;
};

template<typename T>
auto Slice<T>::size() const noexcept -> std::size_t
{
    return end - begin;
}

template<typename T>
auto Slice<T>::operator[] (SliceIndex i) const noexcept -> Slice<T>
{
    return { this->begin + i.begin, std::min(this->begin + i.end, this->end) };
}


template<typename T>
auto make_slice(std::vector<T> const& v) -> Slice<T>
{
    return { v.data(), v.data() + v.size() };
}


class ParserException : public std::exception {
private:
    std::string m_reason;

public:
    ParserException(std::string reason);

    auto what() const noexcept -> const char*;
};

ParserException::ParserException(std::string reason)
    : m_reason{reason}
{}

auto ParserException::what() const noexcept -> const char*
{
    return m_reason.c_str();
}


class ParserBase {
public:
    virtual ~ParserBase() = default;

protected:
    void do_parse(Slice<u8> data, bool oneshot=false);

    auto parse_data(Slice<u8> data) -> Slice<u8>;
    void parse_data_payload(IptsData const& header, Slice<u8> data);
    auto parse_payload_frame(IptsPayload const& header, Slice<u8> data) -> Slice<u8>;
    void parse_payload_frame_reports(IptsPayloadFrame const& header, Slice<u8> data);
    auto parse_report(IptsPayloadFrame const& header, Slice<u8> data) -> Slice<u8>;
    void parse_report_timestamp(IptsReport const& header, Slice<u8> data);
    void parse_report_heatmap_dim(IptsReport const& header, Slice<u8> data);
    void parse_report_heatmap(IptsReport const& header, Slice<u8> data);

    virtual void on_timestamp(IptsTimestampReport const& ts);
    virtual void on_heatmap_dim(IptsHeatmapDim const& dim);
    virtual void on_heatmap(Slice<u8> const& data);
};


void ParserBase::do_parse(Slice<u8> data, bool oneshot)
{
    if (!data.size()) {
        return;
    }

    do {
        data = parse_data(data);
    } while (data.size() && !oneshot);
}

auto ParserBase::parse_data(Slice<u8> data) -> Slice<u8>
{
    IptsData hdr;
    Slice<u8> pld;

    std::copy(data.begin, data.begin + sizeof(hdr), reinterpret_cast<u8*>(&hdr));
    pld = { data.begin + sizeof(hdr), data.begin + sizeof(hdr) + hdr.size };

    if (data.begin + sizeof(hdr) + hdr.size > data.end)
        throw ParserException{"EOF"};

    switch (hdr.type) {
    case 0x00:
        parse_data_payload(hdr, pld);
        break;

    default:
        break;
    };

    return { pld.end, data.end };
}

void ParserBase::parse_data_payload(IptsData const& header, Slice<u8> data)
{
    IptsPayload hdr;
    Slice<u8> pld;

    std::copy(data.begin, data.begin + sizeof(hdr), reinterpret_cast<u8*>(&hdr));
    pld = { data.begin + sizeof(hdr), data.end };

    for (unsigned int i = 0; i < hdr.frames; ++i) {
        pld = parse_payload_frame(hdr, pld);
    }
}

auto ParserBase::parse_payload_frame(IptsPayload const& header, Slice<u8> data) -> Slice<u8>
{
    IptsPayloadFrame hdr;
    Slice<u8> pld;

    std::copy(data.begin, data.begin + sizeof(hdr), reinterpret_cast<u8*>(&hdr));
    pld = { data.begin + sizeof(hdr), data.begin + sizeof(hdr) + hdr.size };

    if (data.begin + sizeof(hdr) + hdr.size > data.end)
        throw ParserException{"EOF"};

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

void ParserBase::parse_payload_frame_reports(IptsPayloadFrame const& header, Slice<u8> data)
{
    while (data.size() >= sizeof(IptsReport)) {
        data = parse_report(header, data);
    }
}

auto ParserBase::parse_report(IptsPayloadFrame const& header, Slice<u8> data) -> Slice<u8>
{
    IptsReport hdr;
    Slice<u8> pld;

    std::copy(data.begin, data.begin + sizeof(hdr), reinterpret_cast<u8*>(&hdr));
    pld = { data.begin + sizeof(hdr), data.begin + sizeof(hdr) + hdr.size };

    switch (hdr.type)
    {
    case 0x400:
        parse_report_timestamp(hdr, pld);
        break;

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

void ParserBase::parse_report_timestamp(IptsReport const& header, Slice<u8> data)
{
    IptsTimestampReport ts;

    std::copy(data.begin, data.begin + sizeof(ts), reinterpret_cast<u8*>(&ts));

    on_timestamp(ts);
}

void ParserBase::parse_report_heatmap_dim(IptsReport const& header, Slice<u8> data)
{
    IptsHeatmapDim dim;

    std::copy(data.begin, data.begin + sizeof(dim), reinterpret_cast<u8*>(&dim));

    on_heatmap_dim(dim);
}

void ParserBase::parse_report_heatmap(IptsReport const& header, Slice<u8> data)
{
    on_heatmap(data);
}


void ParserBase::on_timestamp(IptsTimestampReport const& ts)
{}

void ParserBase::on_heatmap_dim(IptsHeatmapDim const& dim)
{}

void ParserBase::on_heatmap(Slice<u8> const& dim)
{}

} /* namespace iptsd */
