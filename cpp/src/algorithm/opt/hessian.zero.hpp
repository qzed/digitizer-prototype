/*
 * Optimized version of hessian.hpp. Do not include directly.
 */

#include "algorithm/hessian.hpp"


namespace impl {

template<typename T>
void hessian_zero(container::image<math::mat2s_t<T>>& out, container::image<T> const& in)
{
    // kernels
    auto const& kxx = kernels::sobel3_xx<T>;
    auto const& kyy = kernels::sobel3_yy<T>;
    auto const& kxy = kernels::sobel3_xy<T>;

    // strides for data access
    index_t const s_left      = -1;
    index_t const s_center    =  0;
    index_t const s_right     =  1;
    index_t const s_top       = -in.stride();
    index_t const s_top_left  = s_top + s_left;
    index_t const s_top_right = s_top + s_right;
    index_t const s_bot       = -s_top;
    index_t const s_bot_left  = s_bot + s_left;
    index_t const s_bot_right = s_bot + s_right;

    // strides for kernel access
    index_t const k_top_left  = 0;
    index_t const k_top       = 1;
    index_t const k_top_right = 2;
    index_t const k_left      = 3;
    index_t const k_center    = 4;
    index_t const k_right     = 5;
    index_t const k_bot_left  = 6;
    index_t const k_bot       = 7;
    index_t const k_bot_right = 8;

    // processing...
    index_t i = 0;

    // x = 0, y = 0
    {
        auto h = math::num<math::mat2s_t<T>>::zero;

        h.xx += in[i + s_center] * kxx[k_center];
        h.xy += in[i + s_center] * kxy[k_center];
        h.yy += in[i + s_center] * kyy[k_center];

        h.xx += in[i + s_right] * kxx[k_right];
        h.xy += in[i + s_right] * kxy[k_right];
        h.yy += in[i + s_right] * kyy[k_right];

        h.xx += in[i + s_bot] * kxx[k_bot];
        h.xy += in[i + s_bot] * kxy[k_bot];
        h.yy += in[i + s_bot] * kyy[k_bot];

        h.xx += in[i + s_bot_right] * kxx[k_bot_right];
        h.xy += in[i + s_bot_right] * kxy[k_bot_right];
        h.yy += in[i + s_bot_right] * kyy[k_bot_right];

        out[i] = h;
    }
    ++i;

    // 0 < x < n - 1, y = 0
    for (; i < in.size().x - 1; ++i) {
        auto h = math::num<math::mat2s_t<T>>::zero;

        h.xx += in[i + s_left] * kxx[k_left];
        h.xy += in[i + s_left] * kxy[k_left];
        h.yy += in[i + s_left] * kyy[k_left];

        h.xx += in[i + s_center] * kxx[k_center];
        h.xy += in[i + s_center] * kxy[k_center];
        h.yy += in[i + s_center] * kyy[k_center];

        h.xx += in[i + s_right] * kxx[k_right];
        h.xy += in[i + s_right] * kxy[k_right];
        h.yy += in[i + s_right] * kyy[k_right];

        h.xx += in[i + s_bot_left] * kxx[k_bot_left];
        h.xy += in[i + s_bot_left] * kxy[k_bot_left];
        h.yy += in[i + s_bot_left] * kyy[k_bot_left];

        h.xx += in[i + s_bot] * kxx[k_bot];
        h.xy += in[i + s_bot] * kxy[k_bot];
        h.yy += in[i + s_bot] * kyy[k_bot];

        h.xx += in[i + s_bot_right] * kxx[k_bot_right];
        h.xy += in[i + s_bot_right] * kxy[k_bot_right];
        h.yy += in[i + s_bot_right] * kyy[k_bot_right];

        out[i] = h;
    }

    // x = n - 1, y = 0
    {
        auto h = math::num<math::mat2s_t<T>>::zero;

        h.xx += in[i + s_left] * kxx[k_left];
        h.xy += in[i + s_left] * kxy[k_left];
        h.yy += in[i + s_left] * kyy[k_left];

        h.xx += in[i + s_center] * kxx[k_center];
        h.xy += in[i + s_center] * kxy[k_center];
        h.yy += in[i + s_center] * kyy[k_center];

        h.xx += in[i + s_bot_left] * kxx[k_bot_left];
        h.xy += in[i + s_bot_left] * kxy[k_bot_left];
        h.yy += in[i + s_bot_left] * kyy[k_bot_left];

        h.xx += in[i + s_bot] * kxx[k_bot];
        h.xy += in[i + s_bot] * kxy[k_bot];
        h.yy += in[i + s_bot] * kyy[k_bot];

        out[i] = h;
    }
    ++i;

    // 0 < y < n - 1
    while (i < in.size().x * (in.size().y - 1)) {
        // x = 0
        {
            auto h = math::num<math::mat2s_t<T>>::zero;

            h.xx += in[i + s_top] * kxx[k_top];
            h.xy += in[i + s_top] * kxy[k_top];
            h.yy += in[i + s_top] * kyy[k_top];

            h.xx += in[i + s_top_right] * kxx[k_top_right];
            h.xy += in[i + s_top_right] * kxy[k_top_right];
            h.yy += in[i + s_top_right] * kyy[k_top_right];

            h.xx += in[i + s_center] * kxx[k_center];
            h.xy += in[i + s_center] * kxy[k_center];
            h.yy += in[i + s_center] * kyy[k_center];

            h.xx += in[i + s_right] * kxx[k_right];
            h.xy += in[i + s_right] * kxy[k_right];
            h.yy += in[i + s_right] * kyy[k_right];

            h.xx += in[i + s_bot] * kxx[k_bot];
            h.xy += in[i + s_bot] * kxy[k_bot];
            h.yy += in[i + s_bot] * kyy[k_bot];

            h.xx += in[i + s_bot_right] * kxx[k_bot_right];
            h.xy += in[i + s_bot_right] * kxy[k_bot_right];
            h.yy += in[i + s_bot_right] * kyy[k_bot_right];

            out[i] = h;
        }
        ++i;

        // 0 < x < n - 1
        auto const limit = i + in.size().x - 2;
        for (; i < limit; ++i) {
            auto h = math::num<math::mat2s_t<T>>::zero;

            h.xx += in[i + s_top_left] * kxx[k_top_left];
            h.xy += in[i + s_top_left] * kxy[k_top_left];
            h.yy += in[i + s_top_left] * kyy[k_top_left];

            h.xx += in[i + s_top] * kxx[k_top];
            h.xy += in[i + s_top] * kxy[k_top];
            h.yy += in[i + s_top] * kyy[k_top];

            h.xx += in[i + s_top_right] * kxx[k_top_right];
            h.xy += in[i + s_top_right] * kxy[k_top_right];
            h.yy += in[i + s_top_right] * kyy[k_top_right];

            h.xx += in[i + s_left] * kxx[k_left];
            h.xy += in[i + s_left] * kxy[k_left];
            h.yy += in[i + s_left] * kyy[k_left];

            h.xx += in[i + s_center] * kxx[k_center];
            h.xy += in[i + s_center] * kxy[k_center];
            h.yy += in[i + s_center] * kyy[k_center];

            h.xx += in[i + s_right] * kxx[k_right];
            h.xy += in[i + s_right] * kxy[k_right];
            h.yy += in[i + s_right] * kyy[k_right];

            h.xx += in[i + s_bot_left] * kxx[k_bot_left];
            h.xy += in[i + s_bot_left] * kxy[k_bot_left];
            h.yy += in[i + s_bot_left] * kyy[k_bot_left];

            h.xx += in[i + s_bot] * kxx[k_bot];
            h.xy += in[i + s_bot] * kxy[k_bot];
            h.yy += in[i + s_bot] * kyy[k_bot];

            h.xx += in[i + s_bot_right] * kxx[k_bot_right];
            h.xy += in[i + s_bot_right] * kxy[k_bot_right];
            h.yy += in[i + s_bot_right] * kyy[k_bot_right];

            out[i] = h;
        }

        // x = n - 1
        {
            auto h = math::num<math::mat2s_t<T>>::zero;

            h.xx += in[i + s_top_left] * kxx[k_top_left];
            h.xy += in[i + s_top_left] * kxy[k_top_left];
            h.yy += in[i + s_top_left] * kyy[k_top_left];

            h.xx += in[i + s_top] * kxx[k_top];
            h.xy += in[i + s_top] * kxy[k_top];
            h.yy += in[i + s_top] * kyy[k_top];

            h.xx += in[i + s_left] * kxx[k_left];
            h.xy += in[i + s_left] * kxy[k_left];
            h.yy += in[i + s_left] * kyy[k_left];

            h.xx += in[i + s_center] * kxx[k_center];
            h.xy += in[i + s_center] * kxy[k_center];
            h.yy += in[i + s_center] * kyy[k_center];

            h.xx += in[i + s_bot_left] * kxx[k_bot_left];
            h.xy += in[i + s_bot_left] * kxy[k_bot_left];
            h.yy += in[i + s_bot_left] * kyy[k_bot_left];

            h.xx += in[i + s_bot] * kxx[k_bot];
            h.xy += in[i + s_bot] * kxy[k_bot];
            h.yy += in[i + s_bot] * kyy[k_bot];

            out[i] = h;
        }
        ++i;
    }

    // x = 0, y = n - 1
    {
        auto h = math::num<math::mat2s_t<T>>::zero;

        h.xx += in[i + s_top] * kxx[k_top];
        h.xy += in[i + s_top] * kxy[k_top];
        h.yy += in[i + s_top] * kyy[k_top];

        h.xx += in[i + s_top_right] * kxx[k_top_right];
        h.xy += in[i + s_top_right] * kxy[k_top_right];
        h.yy += in[i + s_top_right] * kyy[k_top_right];

        h.xx += in[i + s_center] * kxx[k_center];
        h.xy += in[i + s_center] * kxy[k_center];
        h.yy += in[i + s_center] * kyy[k_center];

        h.xx += in[i + s_right] * kxx[k_right];
        h.xy += in[i + s_right] * kxy[k_right];
        h.yy += in[i + s_right] * kyy[k_right];

        out[i] = h;
    }
    ++i;

    // 0 < x < n - 1, y = n - 1
    for (; i < in.size().product() - 1; ++i) {
        auto h = math::num<math::mat2s_t<T>>::zero;

        h.xx += in[i + s_top_left] * kxx[k_top_left];
        h.xy += in[i + s_top_left] * kxy[k_top_left];
        h.yy += in[i + s_top_left] * kyy[k_top_left];

        h.xx += in[i + s_top] * kxx[k_top];
        h.xy += in[i + s_top] * kxy[k_top];
        h.yy += in[i + s_top] * kyy[k_top];

        h.xx += in[i + s_top_right] * kxx[k_top_right];
        h.xy += in[i + s_top_right] * kxy[k_top_right];
        h.yy += in[i + s_top_right] * kyy[k_top_right];

        h.xx += in[i + s_left] * kxx[k_left];
        h.xy += in[i + s_left] * kxy[k_left];
        h.yy += in[i + s_left] * kyy[k_left];

        h.xx += in[i + s_center] * kxx[k_center];
        h.xy += in[i + s_center] * kxy[k_center];
        h.yy += in[i + s_center] * kyy[k_center];

        h.xx += in[i + s_right] * kxx[k_right];
        h.xy += in[i + s_right] * kxy[k_right];
        h.yy += in[i + s_right] * kyy[k_right];

        out[i] = h;
    }

    // x = n - 1, y = n - 1
    {
        auto h = math::num<math::mat2s_t<T>>::zero;

        h.xx += in[i + s_top_left] * kxx[k_top_left];
        h.xy += in[i + s_top_left] * kxy[k_top_left];
        h.yy += in[i + s_top_left] * kyy[k_top_left];

        h.xx += in[i + s_top] * kxx[k_top];
        h.xy += in[i + s_top] * kxy[k_top];
        h.yy += in[i + s_top] * kyy[k_top];

        h.xx += in[i + s_left] * kxx[k_left];
        h.xy += in[i + s_left] * kxy[k_left];
        h.yy += in[i + s_left] * kyy[k_left];

        h.xx += in[i + s_center] * kxx[k_center];
        h.xy += in[i + s_center] * kxy[k_center];
        h.yy += in[i + s_center] * kyy[k_center];

        out[i] = h;
    }
}

} /* namespace impl */
