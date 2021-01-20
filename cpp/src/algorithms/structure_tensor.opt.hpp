/*
 * Optimized version of structure_tensor.hpp. Do not include directly.
 */

#include "structure_tensor.hpp"


template<>
void structure_tensor_prep<border::zero, border::zero, f32, 3, 3>(
            image<mat2s<f32>>& out, image<f32> const& in,
            kernel<f32, 3, 3> const& kx, kernel<f32, 3, 3> const& ky)
{
    assert(in.shape() == out.shape());

    // strides for data access
    index const s_left      = -1;
    index const s_center    =  0;
    index const s_right     =  1;
    index const s_top       = -stride(in.shape());
    index const s_top_left  = s_top + s_left;
    index const s_top_right = s_top + s_right;
    index const s_bot       = -s_top;
    index const s_bot_left  = s_bot + s_left;
    index const s_bot_right = s_bot + s_right;

    // strides for kernel access
    index const k_top_left  = 0;
    index const k_top       = 1;
    index const k_top_right = 2;
    index const k_left      = 3;
    index const k_center    = 4;
    index const k_right     = 5;
    index const k_bot_left  = 6;
    index const k_bot       = 7;
    index const k_bot_right = 8;

    // processing...
    index i = 0;

    // x = 0, y = 0
    {
        f32 gx = 0.0f, gy = 0.0f;

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
    for (; i < in.shape().x - 1; ++i) {
        f32 gx = 0.0f, gy = 0.0f;

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
        f32 gx = 0.0f, gy = 0.0f;

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
    while (i < in.shape().x * (in.shape().y - 1)) {
        // x = 0
        {
            f32 gx = 0.0f, gy = 0.0f;

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
        auto const limit = i + in.shape().x - 2;
        for (; i < limit; ++i) {
            f32 gx = 0.0f, gy = 0.0f;

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
            f32 gx = 0.0f, gy = 0.0f;

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
        f32 gx = 0.0f, gy = 0.0f;

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
    for (; i < prod(in.shape()) - 1; ++i) {
        f32 gx = 0.0f, gy = 0.0f;

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
        f32 gx = 0.0f, gy = 0.0f;

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
