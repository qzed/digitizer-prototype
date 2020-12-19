#!/usr/bin/env python
from __future__ import print_function

import ctypes
import enum
import sys

from collections import defaultdict


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class IptsData(ctypes.Structure):
    _pack_ = True
    _fields_ = [
        ('type',     ctypes.c_uint32),
        ('size',     ctypes.c_uint32),
        ('buffer',   ctypes.c_uint32),
        ('reserved', ctypes.c_ubyte * 52),
    ]

    def __str__(self):
        return f"IptsData {{ type={self.type:#04x}, size={self.size}, buffer={self.buffer} }}"


class IptsPayload(ctypes.Structure):
    _pack_ = True
    _fields_ = [
        ('counter',  ctypes.c_uint32),
        ('frames',   ctypes.c_uint32),
        ('reserved', ctypes.c_ubyte * 4),
    ]

    def __str__(self):
        return f"IptsPayload {{ counter={self.counter}, frames={self.frames} }}"


class IptsPayloadFrame(ctypes.Structure):
    _pack_ = True
    _fields_ = [
        ('index',    ctypes.c_uint16),
        ('type',     ctypes.c_uint16),
        ('size',     ctypes.c_uint32),
        ('reserved', ctypes.c_ubyte * 8),
    ]

    def __str__(self):
        return f"IptsPayloadFrame {{ index={self.index}, type={self.type:#04x}, size={self.size} }}"


class IptsReport(ctypes.Structure):
    _pack_ = True
    _fields_ = [
        ('type', ctypes.c_uint16),
        ('size', ctypes.c_uint16),
    ]

    def __str__(self):
        return f"IptsReport {{ type={self.type:#04x}, size={self.size} }}"


class IptsStylusReportS(ctypes.Structure):
    _pack_ = True
    _fields_ = [
        ('elements', ctypes.c_uint8),
        ('reserved', ctypes.c_uint8 * 3),
        ('serial',   ctypes.c_uint32),
    ]

    def __str__(self):
        return f"IptsStylusReportS {{ elements={self.elements}, serial={self.serial:08x} }}"


class IptsStylusReportN(ctypes.Structure):
    _pack_ = True
    _fields_ = [
        ('elements', ctypes.c_uint8),
        ('reserved', ctypes.c_uint8 * 3),
    ]

    def __str__(self):
        return f"IptsStylusReportN {{ elements={self.elements} }}"


class IptsStylusDataV1(ctypes.Structure):
    _pack_ = True
    _fields_ = [
        ('reserved1', ctypes.c_uint8 * 4),
        ('mode',      ctypes.c_uint8),
        ('x',         ctypes.c_uint16),
        ('y',         ctypes.c_uint16),
        ('pressure',  ctypes.c_uint16),
        ('reserved2', ctypes.c_uint8),
    ]

    def __str__(self):
        return f"IptsStylusDataV1 {{ "       \
            + f"mode={self.mode:#04x}, "     \
            + f"x={self.x}, y={self.y}, "    \
            + f"pressure={self.pressure} }}"


class IptsStylusDataV2(ctypes.Structure):
    _pack_ = True
    _fields_ = [
        ('timestamp', ctypes.c_uint16),
        ('mode',      ctypes.c_uint16),
        ('x',         ctypes.c_uint16),
        ('y',         ctypes.c_uint16),
        ('pressure',  ctypes.c_uint16),
        ('altitude',  ctypes.c_uint16),
        ('azimuth',   ctypes.c_uint16),
        ('reserved',  ctypes.c_uint8 * 2),
    ]

    def __str__(self):
        return f"IptsStylusDataV2 {{ "             \
            + f"timestamp={self.timestamp:#04x}, " \
            + f"mode={self.mode:#04x}, "           \
            + f"x={self.x}, y={self.y}, "          \
            + f"pressure={self.pressure}, "        \
            + f"altitude={self.altitude}, "        \
            + f"azimuth={self.azimuth} }}"


class IptsHeatmapDim(ctypes.Structure):
    _pack_ = True
    _fields_ = [
        ('height',   ctypes.c_uint8),
        ('width',    ctypes.c_uint8),
        ('y_min',    ctypes.c_uint8),
        ('y_max',    ctypes.c_uint8),
        ('x_min',    ctypes.c_uint8),
        ('x_max',    ctypes.c_uint8),
        ('z_min',    ctypes.c_uint8),
        ('z_max',    ctypes.c_uint8),
    ]

    def __str__(self):
        return f"IptsHeatmapDim {{ "                        \
            + f"height={self.height}, width={self.width}, " \
            + f"y_min={self.y_min}, y_max={self.y_max}, "   \
            + f"x_min={self.x_min}, x_max={self.x_max}, "   \
            + f"z_min={self.z_min}, z_max={self.z_max} }}"


class LogLevel(enum.IntEnum):
    DBG = 0
    INFO = 1
    WARN = 2
    ERR = 3


class ParserStats:
    _LEVEL_DEBUG = 0,

    def __init__(self):
        self.messages = []
        self.unknowns = defaultdict(lambda: defaultdict(list))

    def msg(self, pos, level, msg):
        self.messages.append((pos, level, msg))

    def dbg(self, pos, msg):
        self.msg(pos, LogLevel.DBG, msg)

    def info(self, pos, msg):
        self.msg(pos, LogLevel.INFO, msg)

    def warn(self, pos, msg):
        self.msg(pos, LogLevel.WARN, msg)

    def err(self, pos, msg):
        self.msg(pos, LogLevel.ERR, msg)

    def unknown_type(self, pos, header):
        self.unknowns[type(header)][header.type] += [pos]

    def get_log(self, lvl=LogLevel.INFO):
        return [(pos, level, message) for (pos, level, message) in self.messages if level >= lvl]


class Parser:
    def __init__(self):
        self.pfn_data = defaultdict(lambda: self._parse_data_unknown)
        self.pfn_data[0x00] = self._parse_data_payload

        self.pfn_pldf = defaultdict(lambda: self._parse_payload_frame_unknown)
        self.pfn_pldf[0x06] = self._parse_payload_frame_reports
        self.pfn_pldf[0x07] = self._parse_payload_frame_reports
        self.pfn_pldf[0x08] = self._parse_payload_frame_reports

        self.pfn_rprt = defaultdict(lambda: self._parse_report_unknown)
        self.pfn_rprt[0x403] = self._parse_report_heatmap_dim
        self.pfn_rprt[0x460] = self._parse_report_stylus_v2s
        self.pfn_rprt[0x461] = self._parse_report_stylus_v2n

        self.stats = ParserStats()

    def print_log(self, lvl=LogLevel.INFO):
        msgs = self.stats.get_log(lvl)
        for (pos, level, msg) in msgs:
            eprint(f"{level.name}: {pos}: {msg}")

    def print_unknowns(self, lvl=LogLevel.WARN):
        for ty, v in self.stats.unknowns.items():
            for id, pos in v.items():
                eprint(f"{lvl.name}: unknown {ty.__name__} type: {id:#04x} (count: {len(pos)})")

    def parse(self, data, silent=False):
        self.pos = 0
        cap = len(data)

        while len(data):
            data = self._parse_data(data)

        assert self.pos == cap

        if not silent:
            self.print_log()
            self.print_unknowns()

    def _parse_data(self, data):
        hdr = IptsData.from_buffer_copy(data)
        pld = data[ctypes.sizeof(hdr):ctypes.sizeof(hdr)+hdr.size]

        self.stats.dbg(self.pos, f"{hdr}")
        self._start_data(hdr, pld)

        self.pos += ctypes.sizeof(hdr)
        self.pfn_data[hdr.type](hdr, pld)

        self._end_data(hdr)

        return data[ctypes.sizeof(hdr)+hdr.size:]

    def _parse_data_payload(self, header, data):
        hdr = IptsPayload.from_buffer_copy(data)
        pld = data[ctypes.sizeof(hdr):]

        self.stats.dbg(self.pos, f"{hdr}")
        self._start_data_payload(hdr, pld)

        self.pos += ctypes.sizeof(hdr)
        for _ in range(hdr.frames):
            pld = self._parse_payload_frame(hdr, pld)

        if pld:
            self.stats.err(self.pos, f"payload: skipped {len(pld)} bytes")
            self.pos += len(pld)

        self._end_data_payload(hdr)

    def _parse_data_unknown(self, header, data):
        self.stats.unknown_type(self.pos, header)
        self.pos += len(data)

    def _parse_payload_frame(self, header, data):
        hdr = IptsPayloadFrame.from_buffer_copy(data)
        pld = data[ctypes.sizeof(hdr):ctypes.sizeof(hdr)+hdr.size]

        self.stats.dbg(self.pos, f"{hdr}")
        self._start_payload_frame(hdr, pld)

        self.pos += ctypes.sizeof(hdr)
        self.pfn_pldf[hdr.type](hdr, pld)

        self._end_payload_frame(hdr)
        return data[ctypes.sizeof(hdr)+hdr.size:]

    def _parse_payload_frame_reports(self, header, data):
        while len(data) >= ctypes.sizeof(IptsReport):
            data = self._parse_report(header, data)

        if data:
            self.stats.err(self.pos, f"report: skipped {len(data)} bytes")
            self.pos += len(data)

    def _parse_payload_frame_unknown(self, header, data):
        self.stats.unknown_type(self.pos, header)
        self.pos += len(data)

    def _parse_report(self, header, data):
        hdr = IptsReport.from_buffer_copy(data)
        pld = data[ctypes.sizeof(hdr):ctypes.sizeof(hdr)+hdr.size]

        self.stats.dbg(self.pos, f"{hdr}")
        self._start_report(hdr, pld)

        self.pos += ctypes.sizeof(hdr)
        self.pfn_rprt[hdr.type](hdr, pld)

        self._end_report(hdr)
        return data[ctypes.sizeof(hdr)+hdr.size:]

    def _parse_report_unknown(self, header, data):
        self.stats.unknown_type(self.pos, header)
        self.pos += len(data)

    def _parse_report_heatmap_dim(self, header, data):
        hdr = IptsHeatmapDim.from_buffer_copy(data)

        if len(data) != ctypes.sizeof(hdr):
            self.stats.err(self.pos, f"heatmap dimension report: invalid size: {len(data)}")

        self._on_heatmap_dim(hdr)
        self.pos += len(data)

    def _parse_report_stylus_v2s(self, header, data):
        hdr = IptsStylusReportS.from_buffer_copy(data)
        pld = data[ctypes.sizeof(hdr):]

        self.stats.dbg(self.pos, f"{hdr}")
        self._start_report_stylus_v2s(hdr, pld)

        self.pos += ctypes.sizeof(hdr)
        for i in range(hdr.elements):
            pld = self._parse_stylus_data_v2(hdr, i, pld, self._on_stylus_data_v2s)

        if pld:
            self.stats.err(self.pos, f"stylus report: skipped {len(pld)} bytes")
            self.pos += len(pld)

        self._end_report_stylus_v2s(hdr)

    def _parse_report_stylus_v2n(self, header, data):
        hdr = IptsStylusReportN.from_buffer_copy(data)
        pld = data[ctypes.sizeof(hdr):]

        self.stats.dbg(self.pos, f"{hdr}")
        self._start_report_stylus_v2n(hdr, pld)

        self.pos += ctypes.sizeof(hdr)
        for i in range(hdr.elements):
            pld = self._parse_stylus_data_v2(hdr, i, pld, self._on_stylus_data_v2n)

        if pld:
            self.stats.err(self.pos, f"stylus report: skipped {len(pld)} bytes")
            self.pos += len(pld)

        self._end_report_stylus_v2n(hdr)

    def _parse_stylus_data_v2(self, hdr, index, pld, fn):
        report = IptsStylusDataV2.from_buffer_copy(pld)

        self.stats.dbg(self.pos, f"{report} (index: {index})")
        fn(hdr, index, report)

        self.pos += ctypes.sizeof(report)
        return pld[ctypes.sizeof(report):]

    def _start_data(self, header, payload):
        pass

    def _end_data(self, header):
        pass

    def _start_data_payload(self, header, payload):
        pass

    def _end_data_payload(self, header):
        pass

    def _start_payload_frame(self, header, payload):
        pass

    def _end_payload_frame(self, header):
        pass

    def _start_report(self, header, payload):
        pass

    def _end_report(self, header):
        pass

    def _start_report_stylus_v2s(self, header, payload):
        pass

    def _end_report_stylus_v2s(self, header):
        pass

    def _start_report_stylus_v2n(self, header, payload):
        pass

    def _end_report_stylus_v2n(self, header):
        pass

    def _on_heatmap_dim(self, dim):
        pass

    def _on_stylus_data_v2s(self, header, index, report):
        pass

    def _on_stylus_data_v2n(self, header, index, report):
        pass
