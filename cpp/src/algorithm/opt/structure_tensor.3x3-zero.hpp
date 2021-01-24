/*
 * Optimized version of structure_tensor.hpp. Do not include directly.
 */

#include "algorithm/structure_tensor.hpp"


namespace impl {

template<typename T>
void structure_tensor_3x3_zero(container::image<math::mat2s_t<T>>& out,
                               container::image<T> const& in,
                               container::kernel<T, 3, 3> const& kx,
                               container::kernel<T, 3, 3> const& ky)
{
    assert(in.size() == out.size());

    // strides for data access
    index_t const s_left      = -1;
    index_t const s_center    =  0;
    index_t const s_right     =  1;
    index_t const s_top       = -stride(in.size());
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
        T gx = math::num<T>::zero;
        T gy = math::num<T>::zero;

        gx += in[i + s_center] * kx[k_center];
        gy += in[i + s_center] * ky[k_center];

        gx += in[i + s_right] * kx[k_right];
        gy += in[i + s_right] * ky[k_right];

        gx += in[i + s_bot] * kx[k_bot];
        gy += in[i + s_bot] * ky[k_bot];

        gx += in[i + s_bot_right] * kx[k_bot_right];
        gy += in[i + s_bot_right] * ky[k_bot_right];

        out[i] = { gx * gx, gx * gy, gy * gy };
    }
    ++i;

    // 0 < x < n - 1, y = 0
    for (; i < in.size().x - 1; ++i) {
        T gx = math::num<T>::zero;
        T gy = math::num<T>::zero;

        gx += in[i + s_left] * kx[k_left];
        gy += in[i + s_left] * ky[k_left];

        gx += in[i + s_center] * kx[k_center];
        gy += in[i + s_center] * ky[k_center];

        gx += in[i + s_right] * kx[k_right];
        gy += in[i + s_right] * ky[k_right];

        gx += in[i + s_bot_left] * kx[k_bot_left];
        gy += in[i + s_bot_left] * ky[k_bot_left];

        gx += in[i + s_bot] * kx[k_bot];
        gy += in[i + s_bot] * ky[k_bot];

        gx += in[i + s_bot_right] * kx[k_bot_right];
        gy += in[i + s_bot_right] * ky[k_bot_right];

        out[i] = { gx * gx, gx * gy, gy * gy };
    }

    // x = n - 1, y = 0
    {
        T gx = math::num<T>::zero;
        T gy = math::num<T>::zero;

        gx += in[i + s_left] * kx[k_left];
        gy += in[i + s_left] * ky[k_left];

        gx += in[i + s_center] * kx[k_center];
        gy += in[i + s_center] * ky[k_center];

        gx += in[i + s_bot_left] * kx[k_bot_left];
        gy += in[i + s_bot_left] * ky[k_bot_left];

        gx += in[i + s_bot] * kx[k_bot];
        gy += in[i + s_bot] * ky[k_bot];

        out[i] = { gx * gx, gx * gy, gy * gy };
    }
    ++i;

    // 0 < y < n - 1
    while (i < in.size().x * (in.size().y - 1)) {
        // x = 0
        {
            T gx = math::num<T>::zero;
            T gy = math::num<T>::zero;

            gx += in[i + s_top] * kx[k_top];
            gy += in[i + s_top] * ky[k_top];

            gx += in[i + s_top_right] * kx[k_top_right];
            gy += in[i + s_top_right] * ky[k_top_right];

            gx += in[i + s_center] * kx[k_center];
            gy += in[i + s_center] * ky[k_center];

            gx += in[i + s_right] * kx[k_right];
            gy += in[i + s_right] * ky[k_right];

            gx += in[i + s_bot] * kx[k_bot];
            gy += in[i + s_bot] * ky[k_bot];

            gx += in[i + s_bot_right] * kx[k_bot_right];
            gy += in[i + s_bot_right] * ky[k_bot_right];

            out[i] = { gx * gx, gx * gy, gy * gy };
        }
        ++i;

        // 0 < x < n - 1
        auto const limit = i + in.size().x - 2;
        for (; i < limit; ++i) {
            T gx = math::num<T>::zero;
            T gy = math::num<T>::zero;

            gx += in[i + s_top_left] * kx[k_top_left];
            gy += in[i + s_top_left] * ky[k_top_left];

            gx += in[i + s_top] * kx[k_top];
            gy += in[i + s_top] * ky[k_top];

            gx += in[i + s_top_right] * kx[k_top_right];
            gy += in[i + s_top_right] * ky[k_top_right];

            gx += in[i + s_left] * kx[k_left];
            gy += in[i + s_left] * ky[k_left];

            gx += in[i + s_center] * kx[k_center];
            gy += in[i + s_center] * ky[k_center];

            gx += in[i + s_right] * kx[k_right];
            gy += in[i + s_right] * ky[k_right];

            gx += in[i + s_bot_left] * kx[k_bot_left];
            gy += in[i + s_bot_left] * ky[k_bot_left];

            gx += in[i + s_bot] * kx[k_bot];
            gy += in[i + s_bot] * ky[k_bot];

            gx += in[i + s_bot_right] * kx[k_bot_right];
            gy += in[i + s_bot_right] * ky[k_bot_right];

            out[i] = { gx * gx, gx * gy, gy * gy };
        }

        // x = n - 1
        {
            T gx = math::num<T>::zero;
            T gy = math::num<T>::zero;

            gx += in[i + s_top_left] * kx[k_top_left];
            gy += in[i + s_top_left] * ky[k_top_left];

            gx += in[i + s_top] * kx[k_top];
            gy += in[i + s_top] * ky[k_top];

            gx += in[i + s_left] * kx[k_left];
            gy += in[i + s_left] * ky[k_left];

            gx += in[i + s_center] * kx[k_center];
            gy += in[i + s_center] * ky[k_center];

            gx += in[i + s_bot_left] * kx[k_bot_left];
            gy += in[i + s_bot_left] * ky[k_bot_left];

            gx += in[i + s_bot] * kx[k_bot];
            gy += in[i + s_bot] * ky[k_bot];

            out[i] = { gx * gx, gx * gy, gy * gy };
        }
        ++i;
    }

    // x = 0, y = n - 1
    {
        T gx = math::num<T>::zero;
        T gy = math::num<T>::zero;

        gx += in[i + s_top] * kx[k_top];
        gy += in[i + s_top] * ky[k_top];

        gx += in[i + s_top_right] * kx[k_top_right];
        gy += in[i + s_top_right] * ky[k_top_right];

        gx += in[i + s_center] * kx[k_center];
        gy += in[i + s_center] * ky[k_center];

        gx += in[i + s_right] * kx[k_right];
        gy += in[i + s_right] * ky[k_right];

        out[i] = { gx * gx, gx * gy, gy * gy };
    }
    ++i;

    // 0 < x < n - 1, y = n - 1
    for (; i < in.size().product() - 1; ++i) {
        T gx = math::num<T>::zero;
        T gy = math::num<T>::zero;

        gx += in[i + s_top_left] * kx[k_top_left];
        gy += in[i + s_top_left] * ky[k_top_left];

        gx += in[i + s_top] * kx[k_top];
        gy += in[i + s_top] * ky[k_top];

        gx += in[i + s_top_right] * kx[k_top_right];
        gy += in[i + s_top_right] * ky[k_top_right];

        gx += in[i + s_left] * kx[k_left];
        gy += in[i + s_left] * ky[k_left];

        gx += in[i + s_center] * kx[k_center];
        gy += in[i + s_center] * ky[k_center];

        gx += in[i + s_right] * kx[k_right];
        gy += in[i + s_right] * ky[k_right];

        out[i] = { gx * gx, gx * gy, gy * gy };
    }

    // x = n - 1, y = n - 1
    {
        T gx = math::num<T>::zero;
        T gy = math::num<T>::zero;

        gx += in[i + s_top_left] * kx[k_top_left];
        gy += in[i + s_top_left] * ky[k_top_left];

        gx += in[i + s_top] * kx[k_top];
        gy += in[i + s_top] * ky[k_top];

        gx += in[i + s_left] * kx[k_left];
        gy += in[i + s_left] * ky[k_left];

        gx += in[i + s_center] * kx[k_center];
        gy += in[i + s_center] * ky[k_center];

        out[i] = { gx * gx, gx * gy, gy * gy };
    }
}

} /* namespace impl */
