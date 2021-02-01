#include "parser.hpp"
#include "types.hpp"

#include "control.h"
#include "utils.h"

#include "processor.hpp"
#include "parser.hpp"
#include "types.hpp"
#include "visualization.hpp"

#include "container/image.hpp"

#include "eval/perf.hpp"

#include <cairo/cairo.h>
#include <gtk/gtk.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>


class parser : public parser_base {
public:
    parser(index2_t size);

    auto parse(slice<u8> data) -> container::image<f32> const&;

protected:
    virtual void on_heatmap_dim(ipts_heatmap_dim const& dim);
    virtual void on_heatmap(slice<u8> const& data);

private:
    ipts_heatmap_dim      m_dim;
    container::image<f32> m_img;
};

parser::parser(index2_t size)
    : m_dim{}
    , m_img { size }
{}

auto parser::parse(slice<u8> data) -> container::image<f32> const&
{
    this->do_parse(data, true);
    return m_img;
}

void parser::on_heatmap_dim(ipts_heatmap_dim const& dim)
{
    m_dim = dim;
}

void parser::on_heatmap(slice<u8> const& data)
{
    if (m_dim.width != m_img.size().x || m_dim.height != m_img.size().y) {
        std::cout << "error: invalid heatmap size" << std::endl;
        abort();
    }

    std::transform(data.begin, data.end, m_img.begin(), [&](auto v) {
        return 1.0f - static_cast<f32>(v - m_dim.z_min) / static_cast<f32>(m_dim.z_max - m_dim.z_min);
    });
}


class main_context {
public:
    main_context(GtkWidget* widget, index2_t img_size);

    void submit(container::image<f32> const& img, std::vector<touch_point> const& tps);

    auto draw_event(gfx::cairo::cairo& cr) -> bool;

public:
    static auto on_draw_event(GtkWidget *widget, cairo_t *cr, gpointer user_data) -> gboolean;

private:
    GtkWidget* m_widget;

    visualization m_vis;

    container::image<f32> m_img1;
    container::image<f32> m_img2;
    std::vector<touch_point> m_tps1;
    std::vector<touch_point> m_tps2;

    std::mutex m_lock;

    container::image<f32>* m_img_frnt;
    container::image<f32>* m_img_back;
    std::vector<touch_point>* m_tps_frnt;
    std::vector<touch_point>* m_tps_back;
    bool m_swap;
};

main_context::main_context(GtkWidget* widget, index2_t img_size)
    : m_widget{widget}
    , m_vis{img_size}
    , m_img1{img_size}
    , m_img2{img_size}
    , m_tps1{}
    , m_tps2{}
    , m_img_frnt{&m_img1}
    , m_img_back{&m_img2}
    , m_tps_frnt{&m_tps1}
    , m_tps_back{&m_tps2}
{}

void main_context::submit(container::image<f32> const& img, std::vector<touch_point> const& tps)
{
    {   // set swap to false to prevent read-access in draw
        auto guard = std::lock_guard(m_lock);
        m_swap = false;
    }

    // copy to back-buffer
    *m_img_back = img;
    *m_tps_back = tps;

    {   // set swap to true to indicate that new data has arrived
        auto guard = std::lock_guard(m_lock);
        m_swap = true;
    }

    // request update
    gtk_widget_queue_draw(m_widget);
}

auto main_context::draw_event(gfx::cairo::cairo& cr) -> bool
{
    auto const width  = gtk_widget_get_allocated_width(m_widget);
    auto const height = gtk_widget_get_allocated_height(m_widget);

    {   // check and swap buffers, if necessary
        auto guard = std::lock_guard(m_lock);

        if (m_swap) {
            std::swap(m_img_frnt, m_img_back);
            std::swap(m_tps_frnt, m_tps_back);
            m_swap = false;
        }
    }

    m_vis.draw(cr, *m_img_frnt, *m_tps_frnt, width, height);
    return false;
}

auto main_context::on_draw_event(GtkWidget *widget, cairo_t *cr, gpointer user_data) -> gboolean
{
    auto ctx = reinterpret_cast<main_context*>(user_data);
    auto crw = gfx::cairo::cairo::wrap_raw(cr);

    return ctx->draw_event(crw);
}


auto main(int argc, char** argv) -> int
{
    GtkWidget* window;
    GtkWidget* darea;

    gtk_init(&argc, &argv);

    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

    darea = gtk_drawing_area_new();
    gtk_container_add(GTK_CONTAINER (window), darea);

    auto const size = index2_t { 72, 48 };
    auto ctx = main_context { darea, size };
    auto prc = touch_processor { size };

    auto ctrl = iptsd_control {};
    iptsd_control_start(&ctrl);

    g_signal_connect(G_OBJECT(darea), "draw", G_CALLBACK(main_context::on_draw_event), &ctx);
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    // fix aspect to 3-to-2
    auto geom = GdkGeometry { 0, 0, 0, 0, 0, 0, 0, 0, 1.5f, 1.5f, GDK_GRAVITY_CENTER };

    gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
    gtk_window_set_default_size(GTK_WINDOW(window), 900, 600);
    gtk_window_set_title(GTK_WINDOW(window), "IPTS Processor Prototype");
    gtk_window_set_geometry_hints(GTK_WINDOW(window), NULL, &geom, GDK_HINT_ASPECT);

    gtk_widget_show_all(window);

    auto run = std::atomic_bool(true);

    auto updt = std::thread([&]() -> void {
        using namespace std::chrono_literals;

        auto p = parser { size };
        auto buf = std::vector<u8>(ctrl.device_info.buffer_size);

        while(run.load()) {
            int64_t doorbell = iptsd_control_doorbell(&ctrl);
            if (doorbell < 0) {
                std::cout << "failed to read IPTS doorbell: " << doorbell << std::endl;
                return;
            }

            int size = ctrl.device_info.buffer_size;

            while (doorbell > ctrl.current_doorbell && run.load()) {
                int ret = iptsd_control_read(&ctrl, buf.data(), size);
                if (ret < 0) {
                    std::cout << "failed to read IPTS data: " << ret << std::endl;
                    return;
                }

                auto hm = p.parse({ buf.data(), buf.data() + buf.size() });
                ctx.submit(hm, prc.process(hm));

                ret = iptsd_control_send_feedback(&ctrl);
                if (ret < 0) {
                    std::cout << "failed to send IPTS feedback: " << ret << std::endl;
                    return;
                }
            }

            std::this_thread::sleep_for(10ms);
        }
    });

    gtk_main();

    iptsd_control_stop(&ctrl);

    // TODO: should probably hook into destroy event to stop thread before gtk_main() returns

    run.store(false);
    updt.join();
}
