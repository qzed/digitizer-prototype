#pragma once

#include "processor.hpp"

#include "types.hpp"

#include "gfx/cairo.hpp"
#include "gfx/color.hpp"

#include "container/image.hpp"

#include <vector>


namespace iptsd {

class Visualization {
public:
    Visualization(index2_t heatmap_size);

    void draw(gfx::cairo::Cairo& cr, Image<f32> const& img,
              std::vector<TouchPoint> const& tps, int width, int height);

private:
    Image<gfx::Srgb> m_data;
};

} /* namespace iptsd */
