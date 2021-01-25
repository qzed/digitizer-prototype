#include "parser.hpp"
#include "types.hpp"
#include "visualization.hpp"

#include "algorithm/convolution.hpp"
#include "algorithm/distance_transform.hpp"
#include "algorithm/gaussian_fitting.hpp"
#include "algorithm/hessian.hpp"
#include "algorithm/label.hpp"
#include "algorithm/local_maxima.hpp"
#include "algorithm/structure_tensor.hpp"

#include "container/image.hpp"
#include "container/kernel.hpp"
#include "container/ops.hpp"

#include "eval/perf.hpp"

#include "gfx/cairo.hpp"
#include "gfx/cmap.hpp"

#include "math/num.hpp"
#include "math/vec2.hpp"
#include "math/mat2.hpp"

#include <vector>
#include <numeric>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>


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


class parser : public parser_base {
private:
    std::vector<container::image<f32>> m_data;
    ipts_heatmap_dim m_dim;

public:
    auto parse(char const* file) -> std::vector<container::image<f32>>;

protected:
    virtual void on_heatmap_dim(ipts_heatmap_dim const& dim);
    virtual void on_heatmap(slice<u8> const& data);
};

auto parser::parse(char const* file) -> std::vector<container::image<f32>>
{
    m_data = std::vector<container::image<f32>>{};

    auto const data = read_file(file);
    this->do_parse({data.data(), data.data() + data.size()});

    return std::move(m_data);
}

void parser::on_heatmap_dim(ipts_heatmap_dim const& dim)
{
    m_dim = dim;
}

void parser::on_heatmap(slice<u8> const& data)
{
    auto img = container::image<f32> {{ m_dim.width, m_dim.height }};

    std::transform(data.begin, data.end, img.begin(), [&](auto v) {
        return 1.0f - static_cast<f32>(v - m_dim.z_min) / static_cast<f32>(m_dim.z_max - m_dim.z_min);
    });

    m_data.push_back(img);
}


struct component_stats {
    u32 size;
    f32 volume;
    f32 incoherence;
    u32 maximas;
};


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

    // TODO: parser
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

    auto _test = container::image<f32> {{ 72, 48 }};
    std::fill(_test.begin(), _test.end(), 1.0f);

    auto perf_reg = eval::perf::registry {};
    auto perf_t_total = perf_reg.create_entry("total");
    auto perf_t_prep  = perf_reg.create_entry("preprocessing");
    auto perf_t_st    = perf_reg.create_entry("structure-tensor");
    auto perf_t_stev  = perf_reg.create_entry("structure-tensor.eigenvalues");
    auto perf_t_hess  = perf_reg.create_entry("hessian");
    auto perf_t_rdg   = perf_reg.create_entry("ridge");
    auto perf_t_obj   = perf_reg.create_entry("objective");
    auto perf_t_lmax  = perf_reg.create_entry("objective.maximas");
    auto perf_t_lbl   = perf_reg.create_entry("labels");
    auto perf_t_cscr  = perf_reg.create_entry("component-score");
    auto perf_t_wdt   = perf_reg.create_entry("distance-transform");
    auto perf_t_flt   = perf_reg.create_entry("filter");
    auto perf_t_lmaxf = perf_reg.create_entry("filter.maximas");
    auto perf_t_gfit  = perf_reg.create_entry("gaussian-fitting");

    auto img_pp    = container::image<f32> {{ 72, 48 }};
    auto img_m2_1  = container::image<math::mat2s_t<f32>> { img_pp.size() };
    auto img_m2_2  = container::image<math::mat2s_t<f32>> { img_pp.size() };
    auto img_stev  = container::image<std::array<f32, 2>> { img_pp.size() };
    auto img_rdg   = container::image<f32> { img_pp.size() };
    auto img_obj   = container::image<f32> { img_pp.size() };
    auto img_lbl   = container::image<u16> { img_pp.size() };
    auto img_dm1   = container::image<f32> { img_pp.size() };
    auto img_dm2   = container::image<f32> { img_pp.size() };
    auto img_flt   = container::image<f32> { img_pp.size() };
    auto img_gftmp = container::image<f32> { img_pp.size() };

    auto img_out_color = container::image<gfx::srgba> { img_pp.size() };

    auto kern_pp = alg::conv::kernels::gaussian<f32, 5, 5>(1.0);
    auto kern_st = alg::conv::kernels::gaussian<f32, 5, 5>(1.0);
    auto kern_hs = alg::conv::kernels::gaussian<f32, 5, 5>(1.0);

    auto maximas = std::vector<index_t>{};
    auto cstats = std::vector<component_stats>{};
    auto cscore = std::vector<f32>{};

    auto gfparams = std::vector<gfit::parameters<f32>>{};
    auto gfwindow = index2_t { 11, 11 };
    gfit::reserve(gfparams, 32, gfwindow);

    auto wdt_qbuf = std::vector<alg::wdt::q_item<f32>> {};
    wdt_qbuf.reserve(1024);

    auto wdt_queue = std::priority_queue<alg::wdt::q_item<f32>> { std::less<alg::wdt::q_item<f32>>(), wdt_qbuf };

    auto const heatmaps = parser().parse(argv[2]);

    auto out = std::vector<container::image<f32>>{};
    out.reserve(heatmaps.size());

    auto out_tp = std::vector<std::vector<touch_point>>{};
    out_tp.reserve(heatmaps.size());

    std::cout << "Processing..." << std::endl;

    using duration_t = typename std::chrono::high_resolution_clock::duration;
    auto duration = std::vector<duration_t>{};

    int __i = 0;
    do {
        for (auto const& hm : heatmaps) {
            auto _tr = perf_reg.record(perf_t_total);

            // preprocessing
            {
                auto _r = perf_reg.record(perf_t_prep);

                alg::convolve(img_pp, hm, kern_pp);

                auto const sum = container::ops::sum(img_pp);
                auto const avg = sum / img_pp.size().product();

                container::ops::transform(img_pp, [&](auto const x) {
                    return std::max(x - avg, 0.0f);
                });
            }

            // structure tensor
            {
                auto _r = perf_reg.record(perf_t_st);

                alg::structure_tensor(img_m2_1, img_pp);
                alg::convolve(img_m2_2, img_m2_1, kern_st);
            }

            // eigenvalues of structure tensor
            {
                auto _r = perf_reg.record(perf_t_stev);

                container::ops::transform(img_m2_2, img_stev, [](auto const s) {
                    return s.eigenvalues();
                });
            }

            // hessian
            {
                auto _r = perf_reg.record(perf_t_hess);

                alg::hessian(img_m2_1, img_pp);
                alg::convolve(img_m2_2, img_m2_1, kern_hs);
            }

            // ridge measure
            {
                auto _r = perf_reg.record(perf_t_rdg);

                container::ops::transform(img_m2_2, img_rdg, [](auto h) {
                    auto const [ev1, ev2] = h.eigenvalues();
                    return std::max(ev1, 0.0f) + std::max(ev2, 0.0f);
                });
            }

            // objective for labeling
            {
                auto _r = perf_reg.record(perf_t_obj);

                f32 const wr = 0.9;
                f32 const wh = 1.1;

                for (index_t i = 0; i < img_pp.size().product(); ++i) {
                    img_obj[i] = wh * img_pp[i] - wr * img_rdg[i];
                }
            }

            // local maximas
            {
                auto _r = perf_reg.record(perf_t_lmax);

                maximas.clear();
                find_local_maximas(img_pp, 0.05f, std::back_inserter(maximas));
            }

            // labels
            u16 num_labels;
            {
                auto _r = perf_reg.record(perf_t_lbl);

                num_labels = label<4>(img_lbl, img_obj, 0.0f);
            }

            // component score
            {
                auto _r = perf_reg.record(perf_t_cscr);

                cstats.clear();
                cstats.assign(num_labels, component_stats { 0, 0, 0, 0 });

                for (index_t i = 0; i < img_pp.size().product(); ++i) {
                    auto const label = img_lbl[i];

                    if (label == 0)
                        continue;

                    auto const value = img_pp[i];
                    auto const [ev1, ev2] = img_stev[i];

                    auto const coherence = ev1 + ev2 != 0.0f ? (ev1 - ev2) / (ev1 + ev2) : 1.0;

                    cstats.at(label - 1).size += 1;
                    cstats.at(label - 1).volume += value;
                    cstats.at(label - 1).incoherence += 1.0f - (coherence * coherence);
                }

                for (auto m : maximas) {
                    if (img_lbl[m] > 0) {
                        cstats.at(img_lbl[m] - 1).maximas += 1;
                    }
                }

                cscore.assign(num_labels, 0.0f);
                for (index_t i = 0; i < num_labels; ++i) {
                    auto const c = 100.f;

                    auto const& stats = cstats.at(i);

                    f32 v = c * (stats.incoherence / (stats.size * stats.size))
                        * (stats.maximas > 0 ? 1.0f / stats.maximas : 0.0f);

                    cscore.at(i) = v / (1.0f + v);
                }
            }

            // TODO: limit inclusion to N (e.g. N=16) local maximas by highest inclusion score

            // distance transform
            {
                auto _r = perf_reg.record(perf_t_wdt);

                auto const th_inc = 0.6f;

                auto const wdt_cost = [&](index_t i, index2_t d) -> f32 {
                    f32 const c_dist = 0.1f;
                    f32 const c_ridge = 9.0f;
                    f32 const c_grad = 1.0f;

                    auto const [ev1, ev2] = img_stev[i];
                    auto const grad = std::max(ev1, 0.0f) + std::max(ev2, 0.0f);
                    auto const ridge = img_rdg[i];
                    auto const dist = std::sqrt(static_cast<f32>(d.x * d.x + d.y * d.y));

                    return c_ridge * ridge + c_grad * grad + c_dist * dist;
                };

                auto const wdt_mask = [&](index_t i) -> bool {
                    return img_pp[i] > 0.0f && img_lbl[i] == 0;
                };

                auto const wdt_inc_bin = [&](index_t i) -> bool {
                    return img_lbl[i] > 0 && cscore.at(img_lbl[i] - 1) > th_inc;
                };

                auto const wdt_exc_bin = [&](index_t i) -> bool {
                    return img_lbl[i] > 0 && cscore.at(img_lbl[i] - 1) <= th_inc;
                };

                alg::weighted_distance_transform<4>(img_dm1, wdt_inc_bin, wdt_mask, wdt_cost, wdt_queue, 6.0f);
                alg::weighted_distance_transform<4>(img_dm2, wdt_exc_bin, wdt_mask, wdt_cost, wdt_queue, 6.0f);
            }

            // filter
            {
                auto _r = perf_reg.record(perf_t_flt);

                for (index_t i = 0; i < img_pp.size().product(); ++i) {
                    auto const sigma = 1.0f;

                    // img_out[i] = std::numeric_limits<f32>::max() == img_dm1[i] ? 0.0f : img_dm1[i];

                    auto w_inc = img_dm1[i] / sigma;
                    w_inc = std::exp(-w_inc * w_inc);

                    auto w_exc = img_dm2[i] / sigma;
                    w_exc = std::exp(-w_exc * w_exc);

                    auto const w_total = w_inc + w_exc;
                    auto const w = w_total > 0.0f ? w_inc / w_total : 0.0f;

                    img_flt[i] = img_pp[i] * w;
                }
            }

            // filtered maximas
            {
                auto _r = perf_reg.record(perf_t_lmaxf);

                maximas.clear();
                find_local_maximas(img_flt, 0.05f, std::back_inserter(maximas));
            }

            // gaussian fitting
            if (!maximas.empty()) {
                auto _r = perf_reg.record(perf_t_gfit);

                gfit::reserve(gfparams, maximas.size(), gfwindow);

                for (std::size_t i = 0; i < maximas.size(); ++i) {
                    auto const [x, y] = container::image<f32>::unravel(img_pp.size(), maximas[i]);

                    // TODO: move window inwards instead of clamping?
                    auto const bounds = gfit::bbox {
                        std::max(x - (gfwindow.x - 1) / 2, 0),
                        std::min(x + (gfwindow.x - 1) / 2, img_pp.size().x - 1),
                        std::max(y - (gfwindow.y - 1) / 2, 0),
                        std::min(y + (gfwindow.y - 1) / 2, img_pp.size().y - 1),
                    };

                    gfparams[i].valid  = true;
                    gfparams[i].scale  = 1.0f;
                    gfparams[i].mean   = { static_cast<f32>(x), static_cast<f32>(y) };
                    gfparams[i].prec   = { 1.0f, 0.0f, 1.0f };
                    gfparams[i].bounds = bounds;
                }

                gfit::fit(gfparams, img_flt, img_gftmp, 3);
            } else {
                for (auto& p : gfparams) {
                    p.valid = false;
                }
            }

            _tr.stop();
            out.push_back(hm);

            if (mode == mode_type::plot) {
                out_tp.push_back({});
                out_tp.back().reserve(16);

                for (auto const& p : gfparams) {
                    if (!p.valid) {
                        continue;
                    }

                    auto const x = static_cast<index_t>(p.mean.x);
                    auto const y = static_cast<index_t>(p.mean.y);
                    auto const cs = cscore.at(img_lbl[{ x, y }] - 1);

                    auto const cov = p.prec.inverse();
                    if (!cov.has_value()) {
                        std::cout << "warning: failed to invert matrix\n";
                        continue;
                    }

                    out_tp.back().push_back({ cs, p.scale, p.mean, *cov });
                }
            }
        }
    } while (++__i < 50 && mode == mode_type::perf);

    // statistics
    std::cout << "Performance Statistics:" << std::endl;

    for (auto const e : perf_reg.entries()) {
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

    auto surface = gfx::cairo::image_surface_create(gfx::cairo::format::argb32, { width, height });
    auto cr = gfx::cairo::cairo::create(surface);

    auto vis = visualization {{ 72, 48 }};

    for (std::size_t i = 0; i < out.size(); ++i) {
        vis.draw(cr, out[i], out_tp[i], width, height);

        // write file
        auto fname = std::array<char, 32>{};
        std::snprintf(fname.data(), fname.size(), "out-%04ld.png", i);

        surface.write_to_png(dir_out / fname.data());
    }
}
