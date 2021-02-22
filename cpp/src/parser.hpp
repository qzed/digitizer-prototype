#pragma once

#include "types.hpp"

#include <gsl/span>

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
    void do_parse(gsl::span<const std::byte> data, bool oneshot=false);

    auto parse_data(gsl::span<const std::byte> data) -> gsl::span<const std::byte>;
    void parse_data_payload(IptsData const& header, gsl::span<const std::byte> data);
    auto parse_payload_frame(IptsPayload const& header, gsl::span<const std::byte> data) -> gsl::span<const std::byte>;
    void parse_payload_frame_reports(IptsPayloadFrame const& header, gsl::span<const std::byte> data);
    auto parse_report(IptsPayloadFrame const& header, gsl::span<const std::byte> data) -> gsl::span<const std::byte>;
    void parse_report_timestamp(IptsReport const& header, gsl::span<const std::byte> data);
    void parse_report_heatmap_dim(IptsReport const& header, gsl::span<const std::byte> data);
    void parse_report_heatmap(IptsReport const& header, gsl::span<const std::byte> data);

    virtual void on_timestamp(IptsTimestampReport const& ts);
    virtual void on_heatmap_dim(IptsHeatmapDim const& dim);
    virtual void on_heatmap(gsl::span<const std::byte> const& data);
};

void ParserBase::do_parse(gsl::span<const std::byte> data, bool oneshot)
{
    if (data.empty()) {
        return;
    }

    do {
        data = parse_data(data);
    } while (!data.empty() && !oneshot);
}

auto ParserBase::parse_data(gsl::span<const std::byte> data) -> gsl::span<const std::byte>
{
    IptsData hdr;
    gsl::span<const std::byte> pld;

    std::copy(data.begin(), data.begin() + sizeof(hdr), reinterpret_cast<std::byte*>(&hdr));

    if (sizeof(hdr) + hdr.size > data.size())
        throw ParserException{"EOF"};

    pld = data.subspan(sizeof(hdr), hdr.size);

    switch (hdr.type) {
    case 0x00:
        parse_data_payload(hdr, pld);
        break;

    default:
        break;
    };

    return data.subspan(sizeof(hdr) + hdr.size);
}

void ParserBase::parse_data_payload(IptsData const& header, gsl::span<const std::byte> data)
{
    IptsPayload hdr;
    gsl::span<const std::byte> pld;

    std::copy(data.begin(), data.begin() + sizeof(hdr), reinterpret_cast<std::byte*>(&hdr));
    pld = data.subspan(sizeof(hdr));

    for (unsigned int i = 0; i < hdr.frames; ++i) {
        pld = parse_payload_frame(hdr, pld);
    }
}

auto ParserBase::parse_payload_frame(IptsPayload const& header, gsl::span<const std::byte> data)
    -> gsl::span<const std::byte>
{
    IptsPayloadFrame hdr;
    gsl::span<const std::byte> pld;

    std::copy(data.begin(), data.begin() + sizeof(hdr), reinterpret_cast<std::byte*>(&hdr));

    if (sizeof(hdr) + hdr.size > data.size())
        throw ParserException{"EOF"};

    pld = data.subspan(sizeof(hdr), hdr.size);

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

    return data.subspan(sizeof(hdr) + hdr.size);
}

void ParserBase::parse_payload_frame_reports(IptsPayloadFrame const& header, gsl::span<const std::byte> data)
{
    while (data.size() >= sizeof(IptsReport)) {
        data = parse_report(header, data);
    }
}

auto ParserBase::parse_report(IptsPayloadFrame const& header, gsl::span<const std::byte> data)
    -> gsl::span<const std::byte>
{
    IptsReport hdr;
    gsl::span<const std::byte> pld;

    std::copy(data.begin(), data.begin() + sizeof(hdr), reinterpret_cast<std::byte*>(&hdr));

    if (sizeof(hdr) + hdr.size > data.size())
        throw ParserException{"EOF"};

    pld = data.subspan(sizeof(hdr), hdr.size);

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

    return data.subspan(sizeof(hdr) + hdr.size);
}

void ParserBase::parse_report_timestamp(IptsReport const& header, gsl::span<const std::byte> data)
{
    IptsTimestampReport ts;

    std::copy(data.begin(), data.begin() + sizeof(ts), reinterpret_cast<std::byte*>(&ts));

    on_timestamp(ts);
}

void ParserBase::parse_report_heatmap_dim(IptsReport const& header, gsl::span<const std::byte> data)
{
    IptsHeatmapDim dim;

    std::copy(data.begin(), data.begin() + sizeof(dim), reinterpret_cast<std::byte*>(&dim));

    on_heatmap_dim(dim);
}

void ParserBase::parse_report_heatmap(IptsReport const& header, gsl::span<const std::byte> data)
{
    on_heatmap(data);
}


void ParserBase::on_timestamp(IptsTimestampReport const& ts)
{}

void ParserBase::on_heatmap_dim(IptsHeatmapDim const& dim)
{}

void ParserBase::on_heatmap(gsl::span<const std::byte> const& dim)
{}

} /* namespace iptsd */
