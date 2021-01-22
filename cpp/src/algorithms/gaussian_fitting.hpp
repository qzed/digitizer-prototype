#pragma once

#include "../math.hpp"
#include "../utils/access.hpp"

#include <array>


namespace math {

template<class T>
struct mat6 {
    std::array<T, 6 * 6> data;

    constexpr static auto identity() -> mat6<T>;

    constexpr auto operator[] (index2 i) -> T&;
    constexpr auto operator[] (index2 i) const -> T const&;
};

template<class T>
struct vec6 {
    std::array<T, 6> data;

    constexpr auto operator[] (index i) -> T&;
    constexpr auto operator[] (index i) const -> T const&;
};


template<class T>
inline constexpr auto mat6<T>::identity() -> mat6<T>
{
    auto const _0 = static_cast<T>(0);
    auto const _1 = static_cast<T>(1);

    return {
        _1, _0, _0, _0, _0, _0,
        _0, _1, _0, _0, _0, _0,
        _0, _0, _1, _0, _0, _0,
        _0, _0, _0, _1, _0, _0,
        _0, _0, _0, _0, _1, _0,
        _0, _0, _0, _0, _0, _1,
    };
}

template<typename T>
inline constexpr auto mat6<T>::operator[] (index2 i) -> T&
{
    return utils::access::access(data, i.x * 6 + i.y,
                                 i.x >= 0 && i.x < 6 && i.y >= 0 && i.y < 6,
                                 "invalid matrix access");
}

template<typename T>
inline constexpr auto mat6<T>::operator[] (index2 i) const -> T const&
{
    return utils::access::access(data, i.x * 6 + i.y,
                                 i.x >= 0 && i.x < 6 && i.y >= 0 && i.y < 6,
                                 "invalid matrix access");
}


template<typename T>
inline constexpr auto vec6<T>::operator[] (index i) -> T&
{
    return utils::access::access(data, i);
}

template<typename T>
inline constexpr auto vec6<T>::operator[] (index i) const -> T const&
{
    return utils::access::access(data, i);
}


/**
 * lu_decom() - Perform an LU-decomposition with partial pivoting.
 * @m:  The matrix to factorize.
 * @lu: The result of the factorization, encoded in a single matrix.
 * @p:  The permutation vector.
 *
 * Performes a LU-decomposition on the given matrix A, such that PA = LU with
 * L being a lower triangular matrix and U being an upper triangular matrix
 * and P being the row-pivoting matrix. The pivoting matrix is encoded as
 * vector, containing the row indices of A in the order in which the occur in
 * PA.
 */
template<class T>
auto lu_decomp(mat6<T> const& a, mat6<T>& lu, vec6<index>& p, T eps) -> bool
{
    // initialization
    lu = a;
    p = { 0, 1, 2, 3, 4, 5 };

    // TODO: optimize/unroll?

    // decomposition
    for (index c = 0; c < 6 - 1; ++c) {
        // partial pivoting for current column:
        // swap row r >= c with largest absolute value at [r, c] (i.e. in column) with row c
        {
            // step 1: find element with largest absolute value in column
            index r = 0;
            T v = zero<T>();

            for (index i = c; i < 6; ++i) {
                auto const vi = std::abs(lu[{i, c}]);

                if (v < vi) {
                    v = vi;
                    r = i;
                }
            }

            // step 1.5: abort if we cannot find a sufficiently large pivot
            if (v <= eps) {
                return false;
            }

            // step 2: permutate, swap row r and c
            if (r != c) {
                for (index i = 0; i < 6; ++i) {
                    std::swap(lu[{r, i}], lu[{c, i}]);      // swap U[r, :] and U[c, :]
                }
                std::swap(p[r], p[c]);
            }
        }

        // LU-decomposition step:
        for (index r = c + 1; r < 6; ++r) {
            // L[r, c] = U[r, c] / U[c, c]
            lu[{r, c}] = lu[{r, c}] / lu[{c, c}];

            // U[r, :] = U[r, :] - (U[r, c] / U[c, c]) * U[c, :]
            for (index k = c + 1; k < 6; ++k) {
                lu[{r, k}] = lu[{r, k}] - lu[{r, c}] * lu[{c, k}];
            }
        }
    }

    // last check for r=5, c=5 because we've skipped that above
    if (std::abs(lu[{5, 5}]) <= eps) {
        return false;
    }

    return true;
}

/**
 * lu_solve() - Solve a system of linear equations via a LU-decomposition.
 * @lu: The L and U matrices, encoded in one matrix.
 * @p:  The permutation vector.
 * @b:  The right-hand-side vector of the system.
 * @x:  The vector to solve for.
 *
 * Solves the system of linear equations LUx = Pb, where L is a lower
 * triangular matrix, U is an upper triangular matrix, and P the permutation
 * matrix encoded by the vector p. Essentially performs two steps, first
 * solving Ly = Pb for a temporary vector y and then Ux = y for the desired x.
 */
template<class T>
void lu_solve(mat6<T> const& lu, vec6<index> const& p, vec6<T> const& b, vec6<T>& x)
{
    // step 0: compute Pb
    auto pb = vec6<T> { b[p[0]], b[p[1]], b[p[2]], b[p[3]], b[p[4]], b[p[5]] };

    // step 1: solve Ly = Pb for y (forward substitution)
    auto y = vec6<T>{};

    y[0] = pb[0];
    y[1] = pb[1] - lu[{1, 0}] * y[0];
    y[2] = pb[2] - lu[{2, 0}] * y[0] - lu[{2, 1}] * y[1];
    y[3] = pb[3] - lu[{3, 0}] * y[0] - lu[{3, 1}] * y[1] - lu[{3, 2}] * y[2];
    y[4] = pb[4] - lu[{4, 0}] * y[0] - lu[{4, 1}] * y[1] - lu[{4, 2}] * y[2] - lu[{4, 3}] * y[3];
    y[5] = pb[5] - lu[{5, 0}] * y[0] - lu[{5, 1}] * y[1] - lu[{5, 2}] * y[2] - lu[{5, 3}] * y[3] - lu[{5, 4}] * y[4];

    // step 2: solve Ux = y for x (backward substitution)
    x[5] = y[5];
    x[5] /= lu[{5, 5}];

    x[4] = y[4] - lu[{4, 5}] * x[5];
    x[4] /= lu[{4, 4}];

    x[3] = y[3] - lu[{3, 5}] * x[5] - lu[{3, 4}] * x[4];
    x[3] /= lu[{3, 3}];

    x[2] = y[2] - lu[{2, 5}] * x[5] - lu[{2, 4}] * x[4] - lu[{2, 3}] * x[3];
    x[2] /= lu[{2, 2}];

    x[1] = y[1] - lu[{1, 5}] * x[5] - lu[{1, 4}] * x[4] - lu[{1, 3}] * x[3] - lu[{1, 2}] * x[2];
    x[1] /= lu[{1, 1}];

    x[0] = y[0] - lu[{0, 5}] * x[5] - lu[{0, 4}] * x[4] - lu[{0, 3}] * x[3] - lu[{0, 2}] * x[2] - lu[{0, 1}] * x[1];
    x[0] /= lu[{0, 0}];
}

/**
 * ge_solve() - Solve a system of linear equations via Gaussian elimination.
 * @a: The system matrix A.
 * @b: The right-hand-side vector b.
 * @x: The vector to solve for.
 *
 * Solves the system of linear equations Ax = b using Gaussian elimination
 * with partial pivoting.
 */
template<class T>
auto ge_solve(mat6<T> a, vec6<T> b, vec6<T>& x, T eps) -> bool
{
    // TODO: optimize/unroll?

    // step 1: Gaussian elimination
    for (index c = 0; c < 6 - 1; ++c) {
        // partial pivoting for current column:
        // swap row r >= c with largest absolute value at [r, c] (i.e. in column) with row c
        {
            // step 1: find element with largest absolute value in column
            index r = 0;
            T v = zero<T>();

            for (index i = c; i < 6; ++i) {
                auto const vi = std::abs(a[{i, c}]);

                if (v < vi) {
                    v = vi;
                    r = i;
                }
            }

            // step 1.5: abort if we cannot find a sufficiently large pivot
            if (v <= eps) {
                return false;
            }

            // step 2: permutate, swap row r and c
            if (r != c) {
                for (index i = c; i < 6; ++i) {
                    std::swap(a[{r, i}], a[{c, i}]);        // swap A[r, :] and A[c, :]
                }
                std::swap(b[r], b[c]);                      // swap b[r] and b[c]
            }
        }

        // Gaussian elimination step
        for (index r = c + 1; r < 6; ++r) {
            auto const v = a[{r, c}] / a[{c, c}];

            // b[r] = b[r] - (A[r, c] / A[c, c]) * b[c]
            b[r] = b[r] - v * b[c];

            // A[r, :] = A[r, :] - (A[r, c] / A[c, c]) * A[c, :]
            for (index k = c + 1; k < 6; ++k) {
                a[{r, k}] = a[{r, k}] - v * a[{c, k}];
            }
        }
    }

    // last check for r=5, c=5 because we've skipped that above
    if (std::abs(a[{5, 5}]) <= eps) {
        return false;
    }

    // step 2: backwards substitution
    x[5] = b[5];
    x[5] /= a[{5, 5}];

    x[4] = b[4] - a[{4, 5}] * x[5];
    x[4] /= a[{4, 4}];

    x[3] = b[3] - a[{3, 5}] * x[5] - a[{3, 4}] * x[4];
    x[3] /= a[{3, 3}];

    x[2] = b[2] - a[{2, 5}] * x[5] - a[{2, 4}] * x[4] - a[{2, 3}] * x[3];
    x[2] /= a[{2, 2}];

    x[1] = b[1] - a[{1, 5}] * x[5] - a[{1, 4}] * x[4] - a[{1, 3}] * x[3] - a[{1, 2}] * x[2];
    x[1] /= a[{1, 1}];

    x[0] = b[0] - a[{0, 5}] * x[5] - a[{0, 4}] * x[4] - a[{0, 3}] * x[3] - a[{0, 2}] * x[2] - a[{0, 1}] * x[1];
    x[0] /= a[{0, 0}];

    return true;
}

} /* namespace math */


namespace gfit {

using math::mat6;
using math::vec6;


template<class T>
inline constexpr auto const range = vec2<T> { static_cast<T>(1), static_cast<T>(1) };


struct bbox {
    index xmin, xmax;
    index ymin, ymax;
};

template<class T>
struct parameters {
    bool        valid;      // flag to invalidate parameters
    T           scale;      // alpha
    vec2<T>     mean;       // mu
    mat2s<T>    prec;       // precision matrix, aka. inverse covariance matrix, aka. sigma^-1
    bbox        bounds;     // local bounds for sampling
    image<T>    weights;    // local weights for sampling
};


namespace impl {

/**
 * gaussian_like() - 2D Gaussian probability density function without normalization.
 * @x:    Position at which to evaluate the function.
 * @mean: Mean of the Gaussian.
 * @prec: Precision matrix, i.e. the invariance of the covariance matrix.
 */
template<class T>
auto gaussian_like(vec2<T> x, vec2<T> mean, mat2s<T> prec) -> T
{
    return std::exp(-xtmx(prec, x - mean) / static_cast<T>(2));
}


template<class T, class S>
inline void assemble_system(mat6<S>& m, vec6<S>& rhs, bbox const& b, image<T> const& data, image<S> const& w)
{
    auto const eps = std::numeric_limits<S>::epsilon();

    auto const scale = vec2<S> {
        static_cast<S>(2) * range<S>.x / static_cast<S>(data.shape().x),
        static_cast<S>(2) * range<S>.y / static_cast<S>(data.shape().y),
    };

    std::fill(m.data.begin(), m.data.end(), zero<S>());
    std::fill(rhs.data.begin(), rhs.data.end(), zero<S>());

    for (index iy = b.ymin; iy <= b.ymax; ++iy) {
        for (index ix = b.xmin; ix <= b.xmax; ++ix) {
            auto const x = static_cast<S>(ix) * scale.x - range<S>.x;
            auto const y = static_cast<S>(iy) * scale.y - range<S>.y;

            auto const d = w[{ix - b.xmin, iy - b.ymin}] * static_cast<S>(data[{ix, iy}]);
            auto const v = std::log(d + eps) * d * d;

            rhs[0] += v * x * x;
            rhs[1] += v * x * y;
            rhs[2] += v * y * y;
            rhs[3] += v * x;
            rhs[4] += v * y;
            rhs[5] += v;

            m[{0, 0}] += d * d * x * x * x * x;
            m[{0, 1}] += d * d * x * x * x * y;
            m[{0, 2}] += d * d * x * x * y * y;
            m[{0, 3}] += d * d * x * x * x;
            m[{0, 4}] += d * d * x * x * y;
            m[{0, 5}] += d * d * x * x;

            m[{1, 0}] += d * d * x * x * x * y;
            m[{1, 1}] += d * d * x * x * y * y;
            m[{1, 2}] += d * d * x * y * y * y;
            m[{1, 3}] += d * d * x * x * y;
            m[{1, 4}] += d * d * x * y * y;
            m[{1, 5}] += d * d * x * y;

            m[{2, 0}] += d * d * x * x * y * y;
            m[{2, 1}] += d * d * x * y * y * y;
            m[{2, 2}] += d * d * y * y * y * y;
            m[{2, 3}] += d * d * x * y * y;
            m[{2, 4}] += d * d * y * y * y;
            m[{2, 5}] += d * d * y * y;

            m[{3, 0}] += d * d * x * x * x;
            m[{3, 1}] += d * d * x * x * y;
            m[{3, 2}] += d * d * x * y * y;
            m[{3, 3}] += d * d * x * x;
            m[{3, 4}] += d * d * x * y;
            m[{3, 5}] += d * d * x;

            m[{4, 0}] += d * d * x * x * y;
            m[{4, 1}] += d * d * x * y * y;
            m[{4, 2}] += d * d * y * y * y;
            m[{4, 3}] += d * d * x * y;
            m[{4, 4}] += d * d * y * y;
            m[{4, 5}] += d * d * y;

            m[{5, 0}] += d * d * x * x;
            m[{5, 1}] += d * d * x * y;
            m[{5, 2}] += d * d * y * y;
            m[{5, 3}] += d * d * x;
            m[{5, 4}] += d * d * y;
            m[{5, 5}] += d * d;
        }
    }

    m[{0, 1}] *= static_cast<S>(2);
    m[{1, 1}] *= static_cast<S>(2);
    m[{2, 1}] *= static_cast<S>(2);
    m[{3, 1}] *= static_cast<S>(2);
    m[{4, 1}] *= static_cast<S>(2);
    m[{5, 1}] *= static_cast<S>(2);
}

template<class T>
bool extract_params(vec6<T> const& chi, T& scale, vec2<T>& mean, mat2s<T>& prec, T eps)
{
    prec.xx = -static_cast<T>(2) * chi[0];
    prec.xy = -static_cast<T>(2) * chi[1];
    prec.yy = -static_cast<T>(2) * chi[2];

    // mu = sigma * b = prec^-1 * B
    auto const d = det(prec);
    if (std::abs(d) <= eps) {
        return false;
    }

    mean.x = (prec.yy * chi[3] - prec.xy * chi[4]) / d;
    mean.y = (prec.xx * chi[4] - prec.xy * chi[3]) / d;

    scale = std::exp(chi[5] + xtmx(prec, mean) / static_cast<T>(2));

    return true;
}


template<class T>
inline void update_weight_maps(std::vector<parameters<T>>& params, image<T>& total)
{
    auto const scale = vec2<T> {
        static_cast<T>(2) * range<T>.x / static_cast<T>(total.shape().x),
        static_cast<T>(2) * range<T>.y / static_cast<T>(total.shape().y),
    };

    std::fill(total.begin(), total.end(), zero<T>());

    // compute individual Gaussians in sample windows
    for (auto& p : params) {
        if (!p.valid) {
            continue;
        }

        for (index iy = p.bounds.ymin; iy <= p.bounds.ymax; ++iy) {
            for (index ix = p.bounds.xmin; ix <= p.bounds.xmax; ++ix) {
                auto const x = static_cast<T>(ix) * scale.x - range<T>.x;
                auto const y = static_cast<T>(iy) * scale.y - range<T>.y;

                auto const v = p.scale * gaussian_like<T>({x, y}, p.mean, p.prec);

                p.weights[{ix - p.bounds.xmin, iy - p.bounds.ymin}] = v;
            }
        }
    }

    // sum up total
    for (auto& p : params) {
        if (!p.valid) {
            continue;
        }

        for (index y = p.bounds.ymin; y <= p.bounds.ymax; ++y) {
            for (index x = p.bounds.xmin; x <= p.bounds.xmax; ++x) {
                total[{x, y}] += p.weights[{x - p.bounds.xmin, y - p.bounds.ymin}];
            }
        }
    }

    // normalize weights
    for (auto& p : params) {
        if (!p.valid) {
            continue;
        }

        for (index y = p.bounds.ymin; y <= p.bounds.ymax; ++y) {
            for (index x = p.bounds.xmin; x <= p.bounds.xmax; ++x) {
                if (total[{x, y}] > static_cast<T>(0)) {
                    p.weights[{x - p.bounds.xmin, y - p.bounds.ymin}] /= total[{x, y}];
                }
            }
        }
    }
}

} /* namespace impl */


// TODO: vector as parameter container is not good... drops image memory when resized

template<class T>
void reserve(std::vector<parameters<T>>& params, std::size_t n, index2 shape)
{
    if (n > params.size()) {
        params.resize(n, parameters<T> {
            false,
            static_cast<T>(1),
            { static_cast<T>(0), static_cast<T>(0) },
            { static_cast<T>(1), static_cast<T>(0), static_cast<T>(1) },
            { 0, -1, 0, -1 },
            image<T> { shape },
        });
    }

    for (auto& p : params) {
        p.valid = false;
    }
}

template<class T, class S>
void fit(std::vector<parameters<S>>& params, image<T> const& data, image<S>& tmp,
         unsigned int n_iter, S eps=static_cast<S>(1e-20))
{
    auto const scale = vec2<S> {
        static_cast<S>(2) * range<S>.x / static_cast<S>(data.shape().x),
        static_cast<S>(2) * range<S>.y / static_cast<S>(data.shape().y),
    };

    // down-scaling
    for (auto& p : params) {
        if (!p.valid) {
            continue;
        }

        // scale and center mean
        p.mean.x = p.mean.x * scale.x - range<S>.x;
        p.mean.y = p.mean.y * scale.y - range<S>.y;

        // scale precision matrix (compute (S * Sigma * S^T)^-1 = S^-T * Prec * S^-1)
        p.prec.xx = p.prec.xx / (scale.x * scale.x);
        p.prec.xy = p.prec.xy / (scale.x * scale.y);
        p.prec.yy = p.prec.yy / (scale.y * scale.y);
    }

    // perform iterations
    for (unsigned int i = 0; i < n_iter; ++i) {
        // update weights
        impl::update_weight_maps(params, tmp);

        // fit individual parameters
        for (auto& p : params) {
            auto sys = mat6<S>{};
            auto rhs = vec6<S>{};
            auto chi = vec6<S>{};

            if (!p.valid) {
                continue;
            }

            // assemble system of linear equations
            impl::assemble_system(sys, rhs, p.bounds, data, p.weights);

            // solve systems
            p.valid = math::ge_solve(sys, rhs, chi, eps);
            if (!p.valid) {
                std::cout << "warning: invalid equation system\n";
                continue;
            }

            // get parameters
            p.valid = impl::extract_params(chi, p.scale, p.mean, p.prec, eps);
            if (!p.valid) {
                std::cout << "warning: parameter extraction failed\n";
            }
        }
    }

    // undo down-scaling
    for (auto& p : params) {
        if (!p.valid) {
            continue;
        }

        // un-scale and re-adjust mean
        p.mean.x = (p.mean.x + range<S>.x) / scale.x;
        p.mean.y = (p.mean.y + range<S>.y) / scale.y;

        // un-scale precision matrix
        p.prec.xx = p.prec.xx * scale.x * scale.x;
        p.prec.xy = p.prec.xy * scale.x * scale.y;
        p.prec.yy = p.prec.yy * scale.y * scale.y;
    }
}

} /* namespace gfit */
