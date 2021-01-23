#include "math.hpp"
#include "parser.hpp"
#include "types.hpp"
#include "kernels.hpp"

#include "algorithms/convolution.hpp"
#include "algorithms/distance_transform.hpp"
#include "algorithms/gaussian_fitting.hpp"
#include "algorithms/hessian.hpp"
#include "algorithms/label.hpp"
#include "algorithms/local_maxima.hpp"
#include "algorithms/structure_tensor.hpp"

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
    std::vector<image<f32>> m_data;
    ipts_heatmap_dim m_dim;

public:
    auto parse(char const* file) -> std::vector<image<f32>>;

protected:
    virtual void on_heatmap_dim(ipts_heatmap_dim const& dim);
    virtual void on_heatmap(slice<u8> const& data);
};

auto parser::parse(char const* file) -> std::vector<image<f32>>
{
    m_data = std::vector<image<f32>>{};

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
    auto img = image<f32> {{ m_dim.width, m_dim.height }};

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

    auto img_pp  = image<f32> {{ 72, 48 }};
    auto img_m2_1 = image<math::mat2s_t<f32>> { img_pp.shape() };
    auto img_m2_2 = image<math::mat2s_t<f32>> { img_pp.shape() };
    auto img_stev = image<math::vec2_t<f32>>  { img_pp.shape() };
    auto img_rdg = image<f32> { img_pp.shape() };
    auto img_obj = image<f32> { img_pp.shape() };
    auto img_lbl = image<u16> { img_pp.shape() };
    auto img_dm1 = image<f32> { img_pp.shape() };
    auto img_dm2 = image<f32> { img_pp.shape() };
    auto img_flt = image<f32> { img_pp.shape() };
    auto img_gftmp = image<f32> { img_pp.shape() };

    auto img_out_color = image<gfx::srgba> { img_pp.shape() };

    auto kern_pp = kernels::gaussian<f32, 5, 5>(1.0);
    auto kern_st = kernels::gaussian<f32, 5, 5>(1.0);
    auto kern_hs = kernels::gaussian<f32, 5, 5>(1.0);

    auto maximas = std::vector<index_t>{};
    auto cstats = std::vector<component_stats>{};
    auto cscore = std::vector<f32>{};

    auto gfparams = std::vector<gfit::parameters<f32>>{};
    auto gfwindow = index2_t { 11, 11 };
    gfit::reserve(gfparams, 32, gfwindow);

    auto wdt_qbuf = std::vector<impl::q_item<f32>> {};
    wdt_qbuf.reserve(1024);

    auto wdt_queue = std::priority_queue<impl::q_item<f32>> { std::less<impl::q_item<f32>>(), wdt_qbuf };

    auto const heatmaps = parser().parse(argv[2]);

    auto out = std::vector<image<f32>>{};
    out.reserve(heatmaps.size());

    auto out_tp = std::vector<std::vector<std::tuple<f32, f32, math::vec2_t<f32>, math::mat2s_t<f32>>>>{};
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

                conv(img_pp, hm, kern_pp);
                sub0(img_pp, average(img_pp));
            }

            // structure tensor
            {
                auto _r = perf_reg.record(perf_t_st);

                structure_tensor(img_m2_1, img_pp);
                conv(img_m2_2, img_m2_1, kern_st);
            }

            // eigenvalues of structure tensor
            {
                auto _r = perf_reg.record(perf_t_stev);

                std::transform(img_m2_2.begin(), img_m2_2.end(), img_stev.begin(), [](auto s) {
                    auto const [ew1, ew2] = s.eigenvalues();
                    return math::vec2_t<f32> { ew1, ew2 };
                });
            }

            // hessian
            {
                auto _r = perf_reg.record(perf_t_hess);

                hessian(img_m2_1, img_pp);
                conv(img_m2_2, img_m2_1, kern_hs);
            }

            // ridge measure
            {
                auto _r = perf_reg.record(perf_t_rdg);

                std::transform(img_m2_2.begin(), img_m2_2.end(), img_rdg.begin(), [](auto h) {
                    auto const [ev1, ev2] = h.eigenvalues();
                    return std::max(ev1, 0.0f) + std::max(ev2, 0.0f);
                });
            }

            // objective for labeling
            {
                auto _r = perf_reg.record(perf_t_obj);

                f32 const wr = 0.9;
                f32 const wh = 1.1;

                for (index_t i = 0; i < img_pp.shape().product(); ++i) {
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

                for (index_t i = 0; i < img_pp.shape().product(); ++i) {
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

                weighted_distance_transform<4>(img_dm1, wdt_inc_bin, wdt_mask, wdt_cost, wdt_queue, 6.0f);
                weighted_distance_transform<4>(img_dm2, wdt_exc_bin, wdt_mask, wdt_cost, wdt_queue, 6.0f);
            }

            // filter
            {
                auto _r = perf_reg.record(perf_t_flt);

                for (index_t i = 0; i < img_pp.shape().product(); ++i) {
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
                    auto const [x, y] = unravel(img_pp.shape(), maximas[i]);

                    // TODO: move window inwards instead of clamping?
                    auto const bounds = gfit::bbox {
                        std::max(x - (gfwindow.x - 1) / 2, 0),
                        std::min(x + (gfwindow.x - 1) / 2, img_pp.shape().x - 1),
                        std::max(y - (gfwindow.y - 1) / 2, 0),
                        std::min(y + (gfwindow.y - 1) / 2, img_pp.shape().y - 1),
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
                    if (p.valid) {
                        auto const x = static_cast<index_t>(p.mean.x);
                        auto const y = static_cast<index_t>(p.mean.y);
                        auto const cs = cscore.at(img_lbl[{ x, y }] - 1);

                        out_tp.back().push_back({ cs, p.scale, p.mean, p.prec });
                    }
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

    auto src = gfx::cairo::image_surface_create(img_out_color);
    auto surface = gfx::cairo::image_surface_create(gfx::cairo::format::argb32, { width, height });
    auto cr = gfx::cairo::cairo::create(surface);

    cr.select_font_face("monospace", gfx::cairo::font_slant::normal, gfx::cairo::font_weight::normal);
    cr.set_font_size(12.0);

    for (std::size_t i = 0; i < out.size(); ++i) {
        auto const& img_out = out[i];

        auto const img_w = static_cast<f64>(img_out.shape().x);
        auto const img_h = static_cast<f64>(img_out.shape().y);

        auto const win_w = static_cast<f64>(width);
        auto const win_h = static_cast<f64>(height);

        auto const t = [&](math::vec2_t<f64> p) -> math::vec2_t<f64> {
            return { p.x * (win_w / img_w), win_h - p.y * (win_h / img_h) };
        };

        // plot
        gfx::cmap::cubehelix(0.1, -0.6, 1.0, 2.0)
                .map_into(img_out_color, img_out, {{ 0.1f, 0.7f }});

        // plot heatmap
        auto m = gfx::cairo::matrix::identity();
        m.translate({ 0.0, img_h });
        m.scale({ img_w / win_w, -img_h / win_h });

        auto p = gfx::cairo::pattern::create_for_surface(src);
        p.set_matrix(m);
        p.set_filter(gfx::cairo::filter::nearest);

        cr.set_source(p);
        cr.rectangle({ 0, 0 }, { win_w, win_h });
        cr.fill();

        auto txtbuf = std::array<char, 32>{};

        // plot touch-points
        for (auto const [confidence, scale, mean, prec] : out_tp[i]) {
            auto const sigma = prec.inverse();

            if (!sigma.has_value()) {
                std::cout << "warning: failed to invert sigma\n";
                continue;
            }

            auto const eigen = sigma.value().eigen();

            // get standard deviation
            auto const nstd = 1.0;
            auto const s1 = nstd * std::sqrt(eigen.w[0]);
            auto const s2 = nstd * std::sqrt(eigen.w[1]);

            // eigenvectors scaled with standard deviation
            auto const v1 = eigen.v[0].cast<f64>() * s1;
            auto const v2 = eigen.v[1].cast<f64>() * s2;

            // standard deviation
            cr.set_source(gfx::srgba { 0.0, 0.0, 0.0, 0.33 });

            cr.move_to(t({ mean.x + 0.5, mean.y + 0.5 }));
            cr.line_to(t({ mean.x + 0.5 + v1.x, mean.y + 0.5 + v1.y }));

            cr.move_to(t({ mean.x + 0.5, mean.y + 0.5 }));
            cr.line_to(t({ mean.x + 0.5 + v2.x, mean.y + 0.5 + v2.y }));

            cr.stroke();

            // mean
            cr.set_source(gfx::srgb { 1.0, 0.0, 0.0 });

            cr.move_to(t({ mean.x + 0.1, mean.y + 0.5 }));
            cr.line_to(t({ mean.x + 0.9, mean.y + 0.5 }));

            cr.move_to(t({ mean.x + 0.5, mean.y + 0.1 }));
            cr.line_to(t({ mean.x + 0.5, mean.y + 0.9 }));

            cr.stroke();

            // standard deviation ellipse
            cr.set_source(gfx::srgb { 1.0, 0.0, 0.0 });

            cr.save();

            cr.translate(t({ mean.x + 0.5, mean.y + 0.5 }));
            cr.rotate(std::atan2(v1.x, v1.y));
            cr.scale({s2 * win_w / img_w, s1 * win_h / img_h});
            cr.arc({ 0.0, 0.0 }, 1.0, 0.0, 2.0 * math::num<f64>::pi);

            cr.restore();
            cr.stroke();

            // stats
            cr.set_source(gfx::srgb { 1.0, 1.0, 1.0 });

            std::snprintf(txtbuf.data(), txtbuf.size(), "c:%.02f", confidence);
            cr.move_to(t({ mean.x - 3.5, mean.y + 3.0 }));
            cr.show_text(txtbuf.data());

            std::snprintf(txtbuf.data(), txtbuf.size(), "a:%.02f", std::max(s1, s2) / std::min(s1, s2));
            cr.move_to(t({ mean.x - 3.5, mean.y + 2.0 }));
            cr.show_text(txtbuf.data());

            std::snprintf(txtbuf.data(), txtbuf.size(), "s:%.02f", scale);
            cr.move_to(t({ mean.x - 3.5, mean.y + 1.0 }));
            cr.show_text(txtbuf.data());
        }

        // write file
        auto fname = std::array<char, 32>{};
        std::snprintf(fname.data(), fname.size(), "out-%04ld.png", i);

        surface.write_to_png(dir_out / fname.data());
    }
}
