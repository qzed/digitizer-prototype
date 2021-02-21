#include "processor.hpp"
#include "parser.hpp"
#include "types.hpp"
#include "visualization.hpp"

#include "container/image.hpp"

#include "eval/perf.hpp"

#include "gfx/cairo.hpp"

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
    std::vector<container::Image<f32>> m_data;
    IptsHeatmapDim m_dim;

public:
    auto parse(char const* file) -> std::vector<container::Image<f32>>;

protected:
    virtual void on_heatmap_dim(IptsHeatmapDim const& dim);
    virtual void on_heatmap(Slice<u8> const& data);
};

auto Parser::parse(char const* file) -> std::vector<container::Image<f32>>
{
    m_data = std::vector<container::Image<f32>>{};

    auto const data = read_file(file);
    this->do_parse({data.data(), data.data() + data.size()});

    return std::move(m_data);
}

void Parser::on_heatmap_dim(IptsHeatmapDim const& dim)
{
    m_dim = dim;
}

void Parser::on_heatmap(Slice<u8> const& data)
{
    auto img = container::Image<f32> {{ m_dim.width, m_dim.height }};

    std::transform(data.begin, data.end, img.begin(), [&](auto v) {
        return 1.0f - static_cast<f32>(v - m_dim.z_min) / static_cast<f32>(m_dim.z_max - m_dim.z_min);
    });

    m_data.push_back(img);
}


void print_usage_and_exit(char const* name)
{
    std::cout << "Usage:\n";
    std::cout << "  "  << name << " plot <ipts-data> <output-directory>\n";
    std::cout << "  "  << name << " perf <ipts-data>\n";
    exit(1);
}

enum class mode_type {
    plot,
    perf,
};

auto main(int argc, char** argv) -> int
{
    using namespace std::string_literals;
    mode_type mode;

    if (argc < 2) {
        print_usage_and_exit(argv[0]);
    }

    if (argv[1] == "plot"s) {
        mode = mode_type::plot;
    } else if (argv[1] == "perf"s) {
        mode = mode_type::perf;
    } else {
        print_usage_and_exit(argv[0]);
    }

    if ((mode == mode_type::plot && argc != 4) || (mode == mode_type::perf && argc != 3)) {
        print_usage_and_exit(argv[0]);
    }

    auto const heatmaps = Parser().parse(argv[2]);

    if (heatmaps.empty()) {
        std::cout << "No touch data found!" << std::endl;
        return 0;
    }

    auto proc = TouchProcessor { heatmaps[0].size() };

    auto out = std::vector<container::Image<f32>>{};
    out.reserve(heatmaps.size());

    auto out_tp = std::vector<std::vector<TouchPoint>>{};
    out_tp.reserve(heatmaps.size());

    std::cout << "Processing..." << std::endl;

    int __i = 0;
    do {
        for (auto const& hm : heatmaps) {
            auto const& tp = proc.process(hm);

            out.push_back(hm);
            out_tp.push_back(tp);
        }
    } while (++__i < 50 && mode == mode_type::perf);

    // statistics
    std::cout << "Performance Statistics:" << std::endl;

    for (auto const e : proc.perf().entries()) {
        using ms = std::chrono::microseconds;

        std::cout << "  " << e.name << "\n";
        std::cout << "    N:      " << std::setw(8) << e.n_measurements       << "\n";
        std::cout << "    full:   " << std::setw(8) << e.total<ms>().count()  << "\n";
        std::cout << "    mean:   " << std::setw(8) << e.mean<ms>().count()   << "\n";
        std::cout << "    stddev: " << std::setw(8) << e.stddev<ms>().count() << "\n";
        std::cout << "    min:    " << std::setw(8) << e.min<ms>().count()    << "\n";
        std::cout << "    max:    " << std::setw(8) << e.max<ms>().count()    << "\n";
        std::cout << std::endl;
    }

    if (mode == mode_type::perf) {
        return 0;
    }

    // plot
    std::cout << "Plotting..." << std::endl;

    auto const width  = 900;
    auto const height = 600;

    auto const dir_out = std::filesystem::path { argv[3] };
    std::filesystem::create_directories(dir_out);

    auto surface = gfx::cairo::image_surface_create(gfx::cairo::Format::Argb32, { width, height });
    auto cr = gfx::cairo::Cairo::create(surface);

    auto vis = Visualization { heatmaps[0].size() };

    for (std::size_t i = 0; i < out.size(); ++i) {
        vis.draw(cr, out[i], out_tp[i], width, height);

        // write file
        auto fname = std::array<char, 32>{};
        std::snprintf(fname.data(), fname.size(), "out-%04ld.png", i);

        surface.write_to_png(dir_out / fname.data());
    }
}
