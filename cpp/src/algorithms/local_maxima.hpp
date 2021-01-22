#pragma once

#include "../types.hpp"
#include "../math.hpp"


template<int C=8, typename T, typename O>
void find_local_maximas(image<T> const& data, T threshold, O output_iter)
{
    static_assert(C == 4 || C == 8);

    index_t i = 0;

    /*
     * We use the following kernel to compare entries:
     *
     *   [< ] [< ] [< ]
     *   [< ] [  ] [<=]
     *   [<=] [<=] [<=]
     *
     * Half of the entries use "less or equal", the other half "less than" as
     * operators to ensure that we don't either discard any local maximas or
     * report some multiple times.
     */

    // strides
    index_t const s_left       = -1;
    index_t const s_right      =  1;
    index_t const s_top_center = -stride(data.shape());
    index_t const s_top_left   = s_top_center + s_left;
    index_t const s_top_right  = s_top_center + s_right;
    index_t const s_bot_center = stride(data.shape());
    index_t const s_bot_left   = s_bot_center + s_left;
    index_t const s_bot_right  = s_bot_center + s_right;

    // x = 0, y = 0
    if (data[i] > threshold) {
        bool max = true;

        max &= data[i + s_right] <= data[i];
        max &= data[i + s_bot_center] <= data[i];

        if constexpr (C == 8) {
            max &= data[i + s_bot_right] <= data[i];
        }

        if (max) {
            *output_iter++ = i;
        }
    }
    ++i;

    // 0 < x < n - 1, y = 0
    for (; i < data.shape().x - 1; ++i) {
        if (data[i] <= threshold)
            continue;

        bool max = true;

        max &= data[i + s_left] < data[i];
        max &= data[i + s_right] <= data[i];

        if constexpr (C == 8) {
            max &= data[i + s_bot_left] <= data[i];
        }

        max &= data[i + s_bot_center] <= data[i];

        if constexpr (C == 8) {
            max &= data[i + s_bot_right] <= data[i];
        }

        if (max) {
            *output_iter++ = i;
        }
    }

    // x = n - 1, y = 0
    if (data[i] > threshold) {
        bool max = true;

        max &= data[i + s_left] < data[i];

        if constexpr (C == 8) {
            max &= data[i + s_bot_left] <= data[i];
        }

        max &= data[i + s_bot_center] <= data[i];

        if (max) {
            *output_iter++ = i;
        }
    }
    ++i;

    // 0 < y < n - 1
    while (i < data.shape().x * (data.shape().y - 1)) {
        // x = 0
        if (data[i] > threshold) {
            bool max = true;

            max &= data[i + s_right] <= data[i];
            max &= data[i + s_top_center] < data[i];

            if constexpr (C == 8) {
                max &= data[i + s_top_right] < data[i];
            }

            max &= data[i + s_bot_center] <= data[i];

            if constexpr (C == 8) {
                max &= data[i + s_bot_right] <= data[i];
            }

            if (max) {
                *output_iter++ = i;
            }
        }
        ++i;

        // 0 < x < n - 1
        auto const limit = i + data.shape().x - 2;
        for (; i < limit; ++i) {
            if (data[i] <= threshold)
                continue;

            bool max = true;

            max &= data[i + s_left] < data[i];
            max &= data[i + s_right] <= data[i];

            // top left
            if constexpr (C == 8) {
                max &= data[i + s_top_left] < data[i];
            }

            max &= data[i + s_top_center] < data[i];

            if constexpr (C == 8) {
                max &= data[i + s_top_right] < data[i];
                max &= data[i + s_bot_left] <= data[i];
            }

            max &= data[i + s_bot_center] <= data[i];

            if constexpr (C == 8) {
                max &= data[i + s_bot_right] <= data[i];
            }

            if (max) {
                *output_iter++ = i;
            }
        }

        // x = n - 1
        if (data[i] > threshold) {
            bool max = true;

            max &= data[i + s_left] < data[i];

            if constexpr (C == 8) {
                max &= data[i + s_top_left] < data[i];
            }

            max &= data[i + s_top_center] < data[i];

            if constexpr (C == 8) {
                max &= data[i + s_bot_left] <= data[i];
            }

            max &= data[i + s_bot_center] <= data[i];

            if (max) {
                *output_iter++ = i;
            }
        }
        ++i;
    }

    // x = 0, y = n - 1
    if (data[i] > threshold) {
        bool max = true;

        max &= data[i + s_right] <= data[i];
        max &= data[i + s_top_center] < data[i];

        if constexpr (C == 8) {
            max &= data[i + s_top_right] < data[i];
        }

        if (max) {
            *output_iter++ = i;
        }
    }
    ++i;

    // 0 < x < n - 1, y = n - 1
    for (; i < prod(data.shape()) - 1; ++i) {
        if (data[i] <= threshold)
            continue;

        bool max = true;

        max &= data[i + s_left] < data[i];
        max &= data[i + s_right] <= data[i];

        if constexpr (C == 8) {
            max &= data[i + s_top_left] < data[i];
        }

        max &= data[i + s_top_center] < data[i];

        if constexpr (C == 8) {
            max &= data[i + s_top_right] < data[i];
        }

        if (max) {
            *output_iter++ = i;
        }
    }

    // x = n - 1, y = n - 1
    if (data[i] > threshold) {
        bool max = true;

        max &= data[i + s_left] < data[i];

        if constexpr (C == 8) {
            max &= data[i + s_top_left] < data[i];
        }

        max &= data[i + s_top_center] < data[i];

        if (max) {
            *output_iter++ = i;
        }
    }
}
