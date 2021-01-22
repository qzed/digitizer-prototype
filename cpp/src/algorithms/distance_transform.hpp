#pragma once

#include "../types.hpp"
#include "../math.hpp"

#include <queue>
#include <numeric>


namespace impl {

template<typename T>
struct q_item {
    index_t idx;
    T cost;
};

template<typename T>
auto operator== (struct q_item<T> const& a, struct q_item<T> const& b) noexcept -> bool
{
    return a.cost == b.cost;
}

template<typename T>
auto operator!= (struct q_item<T> const& a, struct q_item<T> const& b) noexcept -> bool
{
    return a.cost != b.cost;
}

template<typename T>
auto operator<= (struct q_item<T> const& a, struct q_item<T> const& b) noexcept -> bool
{
    return a.cost <= b.cost;
}

template<typename T>
auto operator>= (struct q_item<T> const& a, struct q_item<T> const& b) noexcept -> bool
{
    return a.cost >= b.cost;
}

template<typename T>
auto operator< (struct q_item<T> const& a, struct q_item<T> const& b) noexcept -> bool
{
    return a.cost < b.cost;
}

template<typename T>
auto operator> (struct q_item<T> const& a, struct q_item<T> const& b) noexcept -> bool
{
    return a.cost > b.cost;
}


template<typename T>
auto is_masked(T& mask, index_t i) -> bool
{
    return !mask(i);
}

template<typename T>
auto is_foreground(T& bin, index_t i) -> bool
{
    return bin(i);
}

template<typename B, typename M>
auto is_compute(B& bin, M& mask, index_t i) -> bool
{
    return !is_foreground(bin, i) && !is_masked(mask, i);
}

template<typename V, typename T>
auto get_cost(T& cost, index_t i, vec2<int> d) -> V
{
    return cost(i, d);
}

template<typename T, typename Q, typename B, typename M, typename C>
inline void evaluate(image<T>& out, Q& queue, B& bin, M& mask, C& cost, index_t i, index_t stride, index2_t dir, T limit)
{
    if (!is_compute(bin, mask, i + stride))
        return;

    auto const c = out[i] + get_cost<T>(cost, i, dir);

    if (c < out[i + stride] && c < limit) {
        queue.push({ i + stride, c });
    }
}

} /* namespace impl */


template<int N=8, typename T, typename F, typename M, typename C, typename Q>
void weighted_distance_transform(image<T>& out, F& bin, M& mask, C& cost, Q& q, T limit=std::numeric_limits<T>::max())
{
    static_assert(N == 4 || N == 8);

    // strides
    index_t const s_left      = -1;
    index_t const s_right     =  1;
    index_t const s_top       = -stride(out.shape());
    index_t const s_top_left  = s_top + s_left;
    index_t const s_top_right = s_top + s_right;
    index_t const s_bot       = stride(out.shape());
    index_t const s_bot_left  = s_bot + s_left;
    index_t const s_bot_right = s_bot + s_right;

    // step 1: initialize output, queue all non-masked pixels
    index_t i = 0;

    // x = 0, y = 0
    if (!impl::is_foreground(bin, i)) {
        out[i] = std::numeric_limits<T>::max();

        if (!impl::is_masked(mask, i)) {
            auto c = std::numeric_limits<T>::max();

            if (impl::is_foreground(bin, i + s_right)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_right, { -1, 0 }));
            }

            if (impl::is_foreground(bin, i + s_bot)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_bot, { 0, -1 }));
            }

            if (N == 8 && impl::is_foreground(bin, i + s_bot_right)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_bot_right, { -1, -1 }));
            }

            if (c < limit) {
                q.push({ i, c });
            }
        }
    } else {
        out[i] = static_cast<T>(0);
    }
    ++i;

    // 0 < x < n - 1, y = 0
    for (; i < out.shape().x - 1; ++i) {
        if (impl::is_foreground(bin, i)) {
            out[i] = static_cast<T>(0);
            continue;
        }

        out[i] = std::numeric_limits<T>::max();

        if (impl::is_masked(mask, i))
            continue;

        auto c = std::numeric_limits<T>::max();

        if (impl::is_foreground(bin, i + s_left)) {
            c = std::min(c, impl::get_cost<T>(cost, i + s_left, { 1, 0 }));
        }

        if (impl::is_foreground(bin, i + s_right)) {
            c = std::min(c, impl::get_cost<T>(cost, i + s_right, { -1, 0 }));
        }

        if (N == 8 && impl::is_foreground(bin, i + s_bot_left)) {
            c = std::min(c, impl::get_cost<T>(cost, i + s_bot_left, { 1, -1 }));
        }

        if (impl::is_foreground(bin, i + s_bot)) {
            c = std::min(c, impl::get_cost<T>(cost, i + s_bot, { 0, -1 }));
        }

        if (N == 8 && impl::is_foreground(bin, i + s_bot_right)) {
            c = std::min(c, impl::get_cost<T>(cost, i + s_bot_right, { -1, -1 }));
        }

        if (c < limit) {
            q.push({ i, c });
        }
    }

    // x = n - 1, y = 0
    if (!impl::is_foreground(bin, i)) {
        out[i] = std::numeric_limits<T>::max();

        if (!impl::is_masked(mask, i)) {
            auto c = std::numeric_limits<T>::max();

            if (impl::is_foreground(bin, i + s_left)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_left, { 1, 0 }));
            }

            if (N == 8 && impl::is_foreground(bin, i + s_bot_left)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_bot_left, { 1, -1 }));
            }

            if (impl::is_foreground(bin, i + s_bot)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_bot, { 0, -1 }));
            }

            if (c < limit) {
                q.push({ i, c });
            }
        }
    } else {
        out[i] = static_cast<T>(0);
    }
    ++i;

    // 0 < y < n - 1
    while (i < out.shape().x * (out.shape().y - 1)) {
        // x = 0
        if (!impl::is_foreground(bin, i)) {
            out[i] = std::numeric_limits<T>::max();

            if (!impl::is_masked(mask, i)) {
                auto c = std::numeric_limits<T>::max();

                if (impl::is_foreground(bin, i + s_right)) {
                    c = std::min(c, impl::get_cost<T>(cost, i + s_right, { -1, 0 }));
                }

                if (impl::is_foreground(bin, i + s_top)) {
                    c = std::min(c, impl::get_cost<T>(cost, i + s_top, { 0, 1 }));
                }

                if (N == 8 && impl::is_foreground(bin, i + s_top_right)) {
                    c = std::min(c, impl::get_cost<T>(cost, i + s_top_right, { -1, 1 }));
                }

                if (impl::is_foreground(bin, i + s_bot)) {
                    c = std::min(c, impl::get_cost<T>(cost, i + s_bot, { 0, -1 }));
                }

                if (N == 8 && impl::is_foreground(bin, i + s_bot_right)) {
                    c = std::min(c, impl::get_cost<T>(cost, i + s_bot_right, { -1, -1 }));
                }

                if (c < limit) {
                    q.push({ i, c });
                }
            }
        } else {
            out[i] = static_cast<T>(0);
        }
        ++i;

        // 0 < x < n - 1
        auto const limit = i + out.shape().x - 2;
        for (; i < limit; ++i) {
            // if this is a foreground pixel, set it to zero and skip the rest
            if (impl::is_foreground(bin, i)) {
                out[i] = static_cast<T>(0);
                continue;
            }

            // initialize all background pixels to maximum
            out[i] = std::numeric_limits<T>::max();

            // don't evaluate pixels that are excluded by mask
            if (impl::is_masked(mask, i))
                continue;

            // compute minimum cost to any neighboring foreground pixel, if available
            auto c = std::numeric_limits<T>::max();

            if (impl::is_foreground(bin, i + s_left)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_left, { 1, 0 }));
            }

            if (impl::is_foreground(bin, i + s_right)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_right, { -1, 0 }));
            }

            if (N == 8 && impl::is_foreground(bin, i + s_top_left)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_top_left, { 1, 1 }));
            }

            if (impl::is_foreground(bin, i + s_top)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_top, { 0, 1 }));
            }

            if (N == 8 && impl::is_foreground(bin, i + s_top_right)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_top_right, { -1, 1 }));
            }

            if (N == 8 && impl::is_foreground(bin, i + s_bot_left)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_bot_left, { 1, -1 }));
            }

            if (impl::is_foreground(bin, i + s_bot)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_bot, { 0, -1 }));
            }

            if (N == 8 && impl::is_foreground(bin, i + s_bot_right)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_bot_right, { -1, -1 }));
            }

            // if we have a finite projected cost, add this pixel
            if (c < limit) {
                q.push({ i, c });
            }
        }

        // x = n - 1
        if (!impl::is_foreground(bin, i)) {
            out[i] = std::numeric_limits<T>::max();

            if (!impl::is_masked(mask, i)) {
                auto c = std::numeric_limits<T>::max();

                if (impl::is_foreground(bin, i + s_left)) {
                    c = std::min(c, impl::get_cost<T>(cost, i + s_left, { 1, 0 }));
                }

                if (N == 8 && impl::is_foreground(bin, i + s_top_left)) {
                    c = std::min(c, impl::get_cost<T>(cost, i + s_top_left, { 1, 1 }));
                }

                if (impl::is_foreground(bin, i + s_top)) {
                    c = std::min(c, impl::get_cost<T>(cost, i + s_top, { 0, 1 }));
                }

                if (N == 8 && impl::is_foreground(bin, i + s_bot_left)) {
                    c = std::min(c, impl::get_cost<T>(cost, i + s_bot_left, { 1, -1 }));
                }

                if (impl::is_foreground(bin, i + s_bot)) {
                    c = std::min(c, impl::get_cost<T>(cost, i + s_bot, { 0, -1 }));
                }

                if (c < limit) {
                    q.push({ i, c });
                }
            }
        } else {
            out[i] = static_cast<T>(0);
        }
        ++i;
    }

    // x = 0, y = n - 1
    if (!impl::is_foreground(bin, i)) {
        out[i] = std::numeric_limits<T>::max();

        if (!impl::is_masked(mask, i)) {
            auto c = std::numeric_limits<T>::max();

            if (impl::is_foreground(bin, i + s_right)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_right, { -1, 0 }));
            }

            if (impl::is_foreground(bin, i + s_top)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_top, { 0, 1 }));
            }

            if (N == 8 && impl::is_foreground(bin, i + s_top_right)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_top_right, { -1, 1 }));
            }

            if (c < limit) {
                q.push({ i, c });
            }
        }
    } else {
        out[i] = static_cast<T>(0);
    }
    ++i;

    // 0 < x < n - 1, y = n - 1
    for (; i < prod(out.shape()) - 1; ++i) {
        if (impl::is_foreground(bin, i)) {
            out[i] = static_cast<T>(0);
            continue;
        }

        out[i] = std::numeric_limits<T>::max();

        if (impl::is_masked(mask, i))
            continue;

        auto c = std::numeric_limits<T>::max();

        if (impl::is_foreground(bin, i + s_left)) {
            c = std::min(c, impl::get_cost<T>(cost, i + s_left, { 1, 0 }));
        }

        if (impl::is_foreground(bin, i + s_right)) {
            c = std::min(c, impl::get_cost<T>(cost, i + s_right, { -1, 0 }));
        }

        if (N == 8 && impl::is_foreground(bin, i + s_top_left)) {
            c = std::min(c, impl::get_cost<T>(cost, i + s_top_left, { 1, 1 }));
        }

        if (impl::is_foreground(bin, i + s_top)) {
            c = std::min(c, impl::get_cost<T>(cost, i + s_top, { 0, 1 }));
        }

        if (N == 8 && impl::is_foreground(bin, i + s_top_right)) {
            c = std::min(c, impl::get_cost<T>(cost, i + s_top_right, { -1, 1 }));
        }

        if (c < limit) {
            q.push({ i, c });
        }
    }

    // x = n - 1, y = n - 1
    if (!impl::is_foreground(bin, i)) {
        out[i] = std::numeric_limits<T>::max();

        if (!impl::is_masked(mask, i)) {
            auto c = std::numeric_limits<T>::max();

            if (impl::is_foreground(bin, i + s_left)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_left, { 1, 0 }));
            }

            if (N == 8 && impl::is_foreground(bin, i + s_top_left)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_top_left, { 1, 1 }));
            }

            if (impl::is_foreground(bin, i + s_top)) {
                c = std::min(c, impl::get_cost<T>(cost, i + s_top, { 0, 1 }));
            }

            if (c < limit) {
                q.push({ i, c });
            }
        }
    } else {
        out[i] = static_cast<T>(0);
    }

    // step 2: while queue is not empty, get next pixel, write down cost, and add neighbors
    while (!q.empty()) {
        // get next pixel and remove it from queue
        impl::q_item<T> pixel = q.top();
        q.pop();

        // check if someone has been here before; if so, skip this one
        if (out[pixel.idx] <= pixel.cost)
            continue;

        // no one has been here before, so we're guaranteed to be on the lowes cost path
        out[pixel.idx] = pixel.cost;

        // evaluate neighbors
        auto const [x, y] = unravel(out.shape(), pixel.idx);

        if (x > 0) {
            impl::evaluate(out, q, bin, mask, cost, pixel.idx, s_left, { -1, 0 }, limit);
        }

        if (x < out.shape().x - 1) {
            impl::evaluate(out, q, bin, mask, cost, pixel.idx, s_right, { 1, 0 }, limit);
        }

        if (y > 0) {
            if (N == 8 && x > 0) {
                impl::evaluate(out, q, bin, mask, cost, pixel.idx, s_top_left, { -1, -1 }, limit);
            }

            impl::evaluate(out, q, bin, mask, cost, pixel.idx, s_top, { 0, -1 }, limit);

            if (N == 8 && x < out.shape().x - 1) {
                impl::evaluate(out, q, bin, mask, cost, pixel.idx, s_top_right, { 1, -1 }, limit);
            }
        }

        if (y < out.shape().y - 1) {
            if (N == 8 && x > 0) {
                impl::evaluate(out, q, bin, mask, cost, pixel.idx, s_bot_left, { -1, 1 }, limit);
            }

            impl::evaluate(out, q, bin, mask, cost, pixel.idx, s_bot, { 0, 1 }, limit);

            if (N == 8 && x < out.shape().x - 1) {
                impl::evaluate(out, q, bin, mask, cost, pixel.idx, s_bot_right, { 1, 1 }, limit);
            }
        }
    }
}
