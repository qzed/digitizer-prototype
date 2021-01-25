#pragma once

#include "processor.hpp"

#include "types.hpp"

#include "gfx/cairo.hpp"
#include "gfx/color.hpp"

#include "container/image.hpp"

#include <vector>


class visualization {
public:
    visualization(index2_t heatmap_size);

    void draw(gfx::cairo::cairo& cr, container::image<f32> const& img,
              std::vector<touch_point> const& tps, int width, int height);

private:
    container::image<gfx::srgb> m_data;
};
