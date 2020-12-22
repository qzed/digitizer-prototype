#!/usr/bin/python
from __future__ import print_function

from utils.libipts import Parser
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import os
import itertools
import datetime
import traceback
import argparse


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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


def get_local_maximas(data, delta=0.05):
    threshold = np.average(data) + delta

    result = []
    for x1, x2 in itertools.product(range(0, data.shape[0]), range(0, data.shape[1])):
        if data[x1, x2] < threshold:
            continue

        ax1, bx1 = max(x1 - 1, 0), min(x1 + 1, data.shape[0] - 1)
        ax2, bx2 = max(x2 - 1, 0), min(x2 + 1, data.shape[1] - 1)

        area = itertools.product(range(ax1, bx1+1), range(ax2, bx2+1))
        if np.all([((data[x1, x2], x1, x2) >= (data[ix1, ix2], ix1, ix2)) for ix1, ix2 in area]):
            result += [np.array([x1, x2])]

    return result


def area(center, delta, n):
    center = np.rint(center).astype(np.int)

    range_x = (max(center[0] - delta[0], 0), (min(center[0] + delta[0], n[0] - 1)))
    range_y = (max(center[1] - delta[1], 0), (min(center[1] + delta[1], n[1] - 1)))

    return (range_x, range_y)


def data_maps(data, params, drange):
    p = np.zeros((len(params), data.shape[0], data.shape[1]))

    for i, (c, mu, sigma) in enumerate(params):
        range_xy = area(mu, drange, data.shape)

        for x1, x2 in itertools.product(range(range_xy[0][0], range_xy[0][1]), range(range_xy[1][0], range_xy[1][1])):
            p[i, x1, x2] = multivariate_normal.pdf(np.array([x1, x2]), mu, sigma)

    s = np.sum(p, axis=0)
    for i in range(p.shape[0]):
        p[i, :, :] = data * np.divide(p[i, :, :], s, where=s!=0)

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
    sigma = np.matrix(np.linalg.inv(sigma_inv))

    mu = sigma * np.matrix(b).T

    alpha = np.exp(c + 0.5 * (mu.T * sigma_inv * mu).item())

    return alpha, np.array(mu).flatten(), sigma


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

    nstd = 1.5

    print("Loading data...")
    with open(args.file_in[0], 'rb') as f:
        data = f.read()

    parser = TouchParser()
    heatmaps = parser.parse(data)

    fig, ax = plt.subplots()

    plt.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(12, 8)

    print("Processing...")
    ims = []
    for i, hm in enumerate(heatmaps):
        elapsed = datetime.datetime.now() - time_start
        print(f"  Frame {i+1}/{len(heatmaps)}, {((i + 1) / len(heatmaps)) * 100:.2f}%, elapsed: {elapsed}")

        hm = np.maximum(hm - np.average(hm), 0.0)

        p = []
        p.append(ax.imshow(hm.T, vmin = 0.0, vmax = 1.0, animated=True))

        try:
            mu_init = get_local_maximas(hm)

            params_init = [(1.0, mu, np.identity(2)) for mu in mu_init]
            params_est = fit_multi(hm, params_init, 3, drange=(4, 4))
            params_est = [(mu, sigma) for (c, mu, sigma) in params_est]

            for (mu, sigma) in params_est:
                p += ax.plot([mu[0], mu[0]], [mu[1] - 0.4, mu[1] + 0.4], linewidth=1, color='red', animated=True)
                p += ax.plot([mu[0] - 0.4, mu[0] + 0.4], [mu[1], mu[1]], linewidth=1, color='red', animated=True)

                if sigma is not None:
                    eigvals, eigvecs = np.linalg.eigh(sigma)

                    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
                    angle = np.arctan2(vy, vx)

                    width, height = 2.0 * nstd * np.sqrt(eigvals)

                    e = Ellipse(xy=mu, width=width, height=height, angle=np.degrees(angle),
                                facecolor='none', edgecolor='red')
                    ax.add_artist(e)
                    p.append(e)

            ims.append(p)

        except:
            traceback.print_exc() 
            ims.append(p)
            break

    print("Writing images...")
    an = animation.ArtistAnimation(fig, ims, interval=16.66, repeat_delay=1000, blit=True)

    file_out = args.file_out[0]
    os.makedirs(os.path.dirname(os.path.realpath(file_out)), exist_ok=True)
    an.save(file_out, writer='imagemagick')


if __name__ == '__main__':
    main()
