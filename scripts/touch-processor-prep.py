#!/usr/bin/python

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

from libipts import Parser

import numpy as np
import scipy.ndimage
import scipy.signal
import diplib as dip

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

import itertools
import datetime
import argparse

import os
import re


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# Sobel operator kernels for gradients (1st order derivative)
SOBEL_X = np.array([
    [1.0, 0.0, -1.0],
    [2.0, 0.0, -2.0],
    [1.0, 0.0, -1.0],
])

SOBEL_Y = SOBEL_X.T

# Sobel operator kernels for Hessian (2nd order derivative)
SOBEL_XX = np.array([
    [1.0, -2.0, 1.0],
    [2.0, -4.0, 2.0],
    [1.0, -2.0, 1.0],
])

SOBEL_YY = SOBEL_XX.T

SOBEL_XY = np.array([
    [ 1.0, 0.0, -1.0],
    [ 0.0, 0.0,  0.0],
    [-1.0, 0.0,  1.0],
])

# Alternative Sobel operator kernels for Hessian (2nd order derivative)
SOBEL_XX_2 = np.array([
    [1.0, 0.0,  -2.0, 0.0, 1.0],
    [4.0, 0.0,  -8.0, 0.0, 4.0],
    [6.0, 0.0, -12.0, 0.0, 6.0],
    [4.0, 0.0,  -8.0, 0.0, 4.0],
    [1.0, 0.0,  -2.0, 0.0, 1.0],
])

SOBEL_YY_2 = SOBEL_XX_2.T

SOBEL_XY_2 = np.array([
    [ 1.0,  2.0, 0.0, -2.0, -1.0],
    [ 2.0,  4.0, 0.0, -4.0, -2.0],
    [ 0.0,  0.0, 0.0,  0.0,  0.0],
    [-2.0, -4.0, 0.0,  4.0,  2.0],
    [-1.0, -2.0, 0.0,  2.0,  1.0],
])


# Structure kernels for labeling
STRUCTURE_4 = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
])

STRUCTURE_8 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])


# Parser to extract touch data
class TouchParser(Parser):
    def __init__(self):
        super().__init__()
        self.dim = None

    def _on_heatmap_dim(self, dim):
        self.dim = dim

    def _on_heatmap(self, data):
        hm = np.frombuffer(data, dtype=np.ubyte)
        hm = hm.reshape((self.dim.height, self.dim.width))
        hm = np.flip(hm, axis=0).T
        hm = (hm.astype(np.float) - self.dim.z_min) / (self.dim.z_max - self.dim.z_min)

        self.data.append(1.0 - hm)

    def parse(self, data):
        self.data = list()
        super().parse(data, silent=True)
        return np.array(self.data)


# Helper function for gradient
def gradient(img):
    vx = scipy.signal.convolve2d(img, SOBEL_X, mode='same')
    vy = scipy.signal.convolve2d(img, SOBEL_Y, mode='same')

    return np.dstack((vx, vy))


# Helper function for hessian
def hessian(img):
    vxx = scipy.signal.convolve2d(img, SOBEL_XX, mode='same')
    vyy = scipy.signal.convolve2d(img, SOBEL_YY, mode='same')
    vxy = scipy.signal.convolve2d(img, SOBEL_XY, mode='same')

    h = np.zeros((*img.shape, 2, 2))
    h[:, :, 0, 0] = vxx
    h[:, :, 1, 1] = vyy
    h[:, :, 0, 1] = h[:, :, 1, 0] = vxy

    return h


# Structure tensor
def structure_tensor(grad, sigma=1.0):
    sxx = grad[:, :, 0]**2
    syy = grad[:, :, 1]**2
    sxy = grad[:, :, 0] * grad[:, :, 1]

    sxx = scipy.ndimage.gaussian_filter(sxx, sigma)
    syy = scipy.ndimage.gaussian_filter(syy, sigma)
    sxy = scipy.ndimage.gaussian_filter(sxy, sigma)

    s = np.zeros((grad.shape[0], grad.shape[1], 2, 2))
    s[:, :, 0, 0] = sxx
    s[:, :, 1, 1] = syy
    s[:, :, 0, 1] = s[:, :, 1, 0] = sxy

    return s


# Smooth hessian
def smooth_hessian(hessian, sigma=1.0):
    hs = np.copy(hessian)

    hs[:, :, 0, 0] = scipy.ndimage.gaussian_filter(hs[:, :, 0, 0], sigma)
    hs[:, :, 1, 1] = scipy.ndimage.gaussian_filter(hs[:, :, 1, 1], sigma)
    hs[:, :, 0, 1] = hs[:, :, 1, 0] = scipy.ndimage.gaussian_filter(hs[:, :, 0, 1], sigma)

    return hs


def get_local_maximas(data, threshold=0.05):
    result = []

    for x2, x1 in itertools.product(range(data.shape[1]), range(data.shape[0])):
        if data[x1, x2] < threshold:
            continue

        ax1, bx1 = max(x1 - 1, 0), min(x1 + 1, data.shape[0] - 1)
        ax2, bx2 = max(x2 - 1, 0), min(x2 + 1, data.shape[1] - 1)

        area = itertools.product(range(ax1, bx1+1), range(ax2, bx2+1))
        if np.all([((data[x1, x2], x1, x2) >= (data[ix1, ix2], ix1, ix2)) for ix1, ix2 in area]):
            result += [np.array([x1, x2])]

    return np.array(result).reshape((-1, 2))


def get_objects(hm, ridge, wr=1.0, pr=1.0, wh=1.0, ph=1.0, th_obj=0.0):
    return (wh * hm**ph - wr * ridge**pr) > th_obj


def generate_mask(hm, labels, include):
    mask = labels == 0

    for lbl in include:
        mask = np.logical_or(mask, labels == lbl)

    return np.logical_and(hm > 0, mask)


def generate_bin(labels, include):
    mask = np.full(labels.shape, True, dtype=np.bool)

    for lbl in include:
        mask = np.logical_and(mask, labels != lbl)

    return mask


def compute_distance_map(hm, labels, cost, include, sigma=1.0, cutoff=1e-10):
    b = generate_bin(labels, include)
    m = generate_mask(hm, labels, include)

    w = dip.GreyWeightedDistanceTransform(cost, b, m)
    w = np.exp(-(w / sigma)**2)
    w[w < cutoff] = 0

    return w


def compute_weights(hm, ews_s, ridge, labels, sets, c_ridge=9.0, c_grad=1.0, c_offs=0.1):
    cost = ridge * c_ridge + np.sum(np.abs(ews_s), axis=-1) * c_grad + c_offs

    weights = np.array([compute_distance_map(hm, labels, cost, s) for s in sets])
    total = np.sum(weights, axis=0)

    return np.divide(weights, total[None, :, :], out=np.zeros_like(weights), where=total != 0)


def component_score(hm, rot, labels, num_labels, scale=100.0):
    lbl_max = np.zeros(num_labels)
    lbl_vol = np.zeros(num_labels)
    lbl_rot = np.zeros(num_labels)

    for x1, x2 in get_local_maximas(hm):
        lbl_max[labels[x1, x2] - 1] += 1

    lbl_max[lbl_max == 0] = np.inf

    for x2, x1 in itertools.product(range(labels.shape[1]), range(labels.shape[0])):
        if labels[x1, x2] == 0:
            continue

        lbl_vol[labels[x1, x2] - 1] += 1
        lbl_rot[labels[x1, x2] - 1] += rot[x1, x2]

    cscore = 100.0 * (lbl_rot / lbl_vol**2) * (1.0 / lbl_max) + np.finfo(np.float).eps
    # cscore = 1.0 / (1.0 + (1.0 / cscore))   # equivalent to sigmoid(log(cscore))
    cscore = cscore / (cscore + 1)            # equivalent to sigmoid(log(cscore)) and the above

    return cscore


def get_component_sets(hm, rot, labels, num_labels, th_inc=0.6):
    cscore = component_score(hm, rot, labels, num_labels)

    set_inc = set(lbl for lbl in range(1, num_labels + 1) if cscore[lbl - 1] > th_inc)
    set_exc = set(range(1, num_labels + 1)) - set_inc

    return set_inc, set_exc


# Pre-smoothing and noise floor removal
def preprocess(hm, sigma=1.0):
    hm = scipy.ndimage.gaussian_filter(hm, sigma)
    hm = np.maximum(hm - np.average(hm), 0.0)
    return hm


def filter(hm):
    ms = get_local_maximas(hm)

    g = gradient(hm)
    s = structure_tensor(g)
    h = smooth_hessian(hessian(hm))

    ews_s, _ = np.linalg.eig(s)
    ews_h, _ = np.linalg.eig(h)

    _cnum = (ews_s[:, :, 0] - ews_s[:, :, 1])
    _cdiv = (ews_s[:, :, 0] + ews_s[:, :, 1])
    coherence = np.divide(_cnum, _cdiv, out=np.full_like(_cnum, 1.0), where=_cdiv!=0)**2
    rot = 1.0 - coherence

    ridge = np.sum(np.maximum(ews_h, 0.0), axis=-1)

    objects = get_objects(hm, ridge)
    labels, num_labels = scipy.ndimage.label(objects, STRUCTURE_4)

    set_inc, set_exc = get_component_sets(hm, rot, labels, num_labels)

    ws = compute_weights(hm, ews_s, ridge, labels, [set_inc, set_exc])

    return ws[0, :, :] * hm


def area(center, delta, n):
    center = np.rint(center).astype(np.int)

    range_x = (max(center[0] - delta[0], 0), (min(center[0] + delta[0] + 1, n[0])))
    range_y = (max(center[1] - delta[1], 0), (min(center[1] + delta[1] + 1, n[1])))

    return (range_x, range_y)


def data_maps(data, params, drange):
    p = np.zeros((len(params), data.shape[0], data.shape[1]))

    for i, (c, mu, sigma_inv) in enumerate(params):
        range_xy = area(mu, drange, data.shape)

        for x1, x2 in itertools.product(range(range_xy[0][0], range_xy[0][1]), range(range_xy[1][0], range_xy[1][1])):
            d = np.matrix([[x1 - mu[0]], [x2 - mu[1]]])
            p[i, x1, x2] = c * np.exp(-0.5 * d.T @ sigma_inv @ d)

    s = np.sum(p, axis=0)
    for i in range(p.shape[0]):
        p[i, :, :] = data * np.divide(p[i, :, :], s, out=np.zeros_like(s), where=s!=0)

    return p


def assemble_system(data, center, drange):
    range_xy = area(center, drange, data.shape)

    system = np.zeros(shape=(6,6))
    rhs = np.zeros(shape=(6,))

    for x1, x2 in itertools.product(range(range_xy[0][0], range_xy[0][1]), range(range_xy[1][0], range_xy[1][1])):
        if data[x1, x2] < 1e-20:
            continue

        base = np.array([x1**2, 2 * x1 * x2, x2**2, x1, x2, 1.0])
        system += np.outer(base, base) * data[x1, x2]**2
        rhs += np.log(data[x1, x2]) * base * data[x1, x2]**2

    return system, rhs


def gaussian_from_sle(a, b, c):
    sigma_inv = -2.0 * a

    mu = np.matrix(np.linalg.inv(sigma_inv)) * np.matrix(b).T

    alpha = np.exp(c + 0.5 * (mu.T * sigma_inv * mu).item())

    return alpha, np.array(mu).flatten(), np.matrix(sigma_inv)


def fit_single(data, mu_init, drange=(2, 2)):
    # assemble and solve system
    system, rhs = assemble_system(data, mu_init, drange=drange)
    chi = np.linalg.solve(system, rhs)

    # extract parameters
    a = np.array([[chi[0], chi[1]], [chi[1], chi[2]]])
    b = chi[3:5]
    c = chi[5]

    # convert to Gaussian and return
    return gaussian_from_sle(a, b, c)


def fit_multi(data, params_init, n_iter, drange=(2, 2)):
    params = params_init

    for i in range(n_iter):
        cdata = data_maps(data, params, drange)

        params_est = []
        for j in range(cdata.shape[0]):
            try:
                mu = params[j][1]
                result = fit_single(cdata[j], mu, drange=drange)

                # only add positive semidefinite
                if np.all(np.linalg.eigvals(result[2]) >= 0):
                    params_est.append(result)
                else:
                    eprint("Warning: matrix not positive semidefinite")
            except np.linalg.LinAlgError as e:
                # likely caused by singular matrix: ignore this datapoint
                eprint(f"Warning: {e}")

        params = params_est

    return params


def main():
    parser = argparse.ArgumentParser(
            description="Process raw IPTS touch data and output image files for prototyping")
    parser.add_argument('file_in', metavar='INPUT', type=str, nargs=1, help='raw IPTS input data')
    parser.add_argument('file_out', metavar='OUTPUT', type=str, nargs=1, help='output image base name')
    args = parser.parse_args()

    time_start = datetime.datetime.now()

    print("Loading data...")
    with open(args.file_in[0], 'rb') as f:
        data = f.read()

    parser = TouchParser()
    heatmaps = parser.parse(data)

    fig, ax = plt.subplots()
    ax.axis('off')

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(12, 8)

    print("Processing...")
    ims = []
    for i, hm in enumerate(heatmaps):
        elapsed = datetime.datetime.now() - time_start
        print(f"  Frame {i+1}/{len(heatmaps)}, {((i + 1) / len(heatmaps)) * 100:.2f}%, elapsed: {elapsed}")

        # process
        hm = hm.T
        hm = preprocess(hm)
        hmf = filter(hm)

        ms = get_local_maximas(hmf)

        params_init = [(1.0, mu, np.identity(2)) for mu in ms]
        params_est = fit_multi(hmf, params_init, 3, drange=(3, 3))
        params_est = [(mu, np.linalg.inv(sigma_inv)) for (c, mu, sigma_inv) in params_est if sigma_inv is not None]

        # plot
        nstd = 1.5

        p = list()
        p.append(ax.imshow(hmf, vmin=0.0, vmax=0.3, animated=True))
        p += ax.contour(hm, levels=[0.01], colors='black').collections
        p += ax.plot(ms[:, 1], ms[:, 0], 'b+', color='black', ms=11)

        for (mu, sigma) in params_est:
            eigvals, eigvecs = np.linalg.eigh(sigma)

            vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
            angle = np.arctan2(vx, vy)

            width, height = 2.0 * nstd * np.sqrt(eigvals)

            aspect = np.min([width, height]) / np.max([width, height])
            if aspect < 0.45:
                continue

            p += ax.plot(mu[1], mu[0], 'b+', ms=10, color='red', animated=True)

            e = Ellipse(xy=(mu[1], mu[0]), width=width, height=height, angle=np.degrees(angle),
                        facecolor='none', edgecolor='red')
            ax.add_artist(e)
            p.append(e)

        ims.append(p)

    print("Writing images...")
    an = animation.ArtistAnimation(fig, ims, interval=16.66, repeat_delay=1000, blit=True)

    file_out = args.file_out[0]
    dir_out = os.path.dirname(os.path.realpath(file_out))
    os.makedirs(dir_out, exist_ok=True)
    an.save(file_out, writer='imagemagick')

    # rename files
    r = re.compile('(.+)-(\d+).(.+)')
    for file in os.listdir(dir_out):
        m = r.match(file)
        os.rename(f"{dir_out}/{file}", f"{dir_out}/{m[1]}-{int(m[2]):04d}.{m[3]}")


if __name__ == '__main__':
    main()
