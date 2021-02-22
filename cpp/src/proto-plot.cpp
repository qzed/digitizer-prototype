#include "processor.hpp"
#include "parser.hpp"
#include "types.hpp"
#include "visualization.hpp"

#include "container/image.hpp"

#include "eval/perf.hpp"

#include "gfx/cairo.hpp"

#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

using namespace iptsd;


auto read_file(char const* path) -> std::vector<u8>
{
    std::ifstream ifs;
    ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    ifs.open(path, std::ios::binary | std::ios::ate);

    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<u8> buffer(size);
    ifs.read(reinterpret_cast<char*>(buffer.data()), size);

    return buffer;
}


class Parser : public ParserBase {
private:
    std::vector<Image<f32>> m_data;
    IptsHeatmapDim m_dim;

public:
    auto parse(char const* file) -> std::vector<Image<f32>>;

protected:
    virtual void on_heatmap_dim(IptsHeatmapDim const& dim);
    virtual void on_heatmap(gsl::span<const std::byte> const& data);
};

auto Parser::parse(char const* file) -> std::vector<Image<f32>>
{
    m_data = std::vector<Image<f32>>{};

    auto const data = read_file(file);
    this->do_parse(gsl::as_bytes(gsl::span{data}));

    return std::move(m_data);
}

void Parser::on_heatmap_dim(IptsHeatmapDim const& dim)
{
    m_dim = dim;
}

void Parser::on_heatmap(gsl::span<const std::byte> const& data)
{
    auto img = Image<f32> {{ m_dim.width, m_dim.height }};

    std::transform(data.begin(), data.end(), img.begin(), [&](auto v) {
        auto const n = static_cast<f32>(m_dim.z_max - m_dim.z_min);
        auto const x = static_cast<f32>(std::to_integer<u8>(v) - m_dim.z_min);

        return 1.0f - x / n;
    });

    m_data.push_back(img);
}


enum class mode_type {
    plot,
    perf,
};

auto main(int argc, char** argv) -> int
{
    spdlog::set_pattern("[%X.%e] [%^%l%$] %v");

    auto mode = mode_type::plot;
    auto path_in = std::string{};
    auto path_out = std::string{};

    auto app = CLI::App { "Digitizer Prototype -- Plotter" };
    app.failure_message(CLI::FailureMessage::help);
    app.set_help_all_flag("--help-all", "Show full help message");
    app.require_subcommand(1);

    auto cmd_plot = app.add_subcommand("plot", "Plot results to PNG files");
    cmd_plot->callback([&]() { mode = mode_type::plot; });
    cmd_plot->add_option("input", path_in, "Input file")->required();
    cmd_plot->add_option("output", path_out, "Output directory")->required();

    auto cmd_perf = app.add_subcommand("perf", "Evaluate performance");
    cmd_perf->callback([&]() { mode = mode_type::perf; });
    cmd_perf->add_option("input", path_in, "Input file")->required();

    CLI11_PARSE(app, argc, argv);

    auto const heatmaps = Parser().parse(path_in.c_str());

    if (heatmaps.empty()) {
        spdlog::warn("No touch data found!");
        return 0;
    }

    auto proc = TouchProcessor { heatmaps[0].size() };

    auto out = std::vector<Image<f32>>{};
    out.reserve(heatmaps.size());

    auto out_tp = std::vector<std::vector<TouchPoint>>{};
    out_tp.reserve(heatmaps.size());

    spdlog::info("Processing...");

    int __i = 0;
    do {
        for (auto const& hm : heatmaps) {
            auto const& tp = proc.process(hm);

            out.push_back(hm);
            out_tp.push_back(tp);
        }
    } while (++__i < 50 && mode == mode_type::perf);

    // statistics
    spdlog::info("Performance Statistics:");

    for (auto const e : proc.perf().entries()) {
        using ms = std::chrono::microseconds;

        spdlog::info("  {}", e.name);
        spdlog::info("    N:      {:8d}", e.n_measurements);
        spdlog::info("    full:   {:8d}", e.total<ms>().count());
        spdlog::info("    mean:   {:8d}", e.mean<ms>().count());
        spdlog::info("    stddev: {:8d}", e.stddev<ms>().count());
        spdlog::info("    min:    {:8d}", e.min<ms>().count());
        spdlog::info("    max:    {:8d}", e.max<ms>().count());
        spdlog::info("");
    }

    if (mode == mode_type::perf) {
        return 0;
    }

    // plot
    spdlog::info("Plotting...");

    auto const width  = 900;
    auto const height = 600;

    auto const dir_out = std::filesystem::path { path_out };
    std::filesystem::create_directories(dir_out);

    auto surface = gfx::cairo::image_surface_create(gfx::cairo::Format::Argb32, { width, height });
    auto cr = gfx::cairo::Cairo::create(surface);

    auto vis = Visualization { heatmaps[0].size() };

    for (std::size_t i = 0; i < out.size(); ++i) {
        vis.draw(cr, out[i], out_tp[i], width, height);

        // write file
        auto fname = std::array<char, 32>{};
        fmt::format_to_n(fname.begin(), fname.size(), "out-{:04d}.png", i);

        surface.write_to_png(dir_out / fname.data());
    }
}
