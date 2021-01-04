#!/usr/bin/python

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

from libipts import Parser

import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

import datetime
import argparse
import re


# Sobel operator kernels for gradients
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


def hessian(img):
    vxx = scipy.signal.convolve2d(img, SOBEL_XX, mode='same')
    vyy = scipy.signal.convolve2d(img, SOBEL_YY, mode='same')
    vxy = scipy.signal.convolve2d(img, SOBEL_XY, mode='same')

    h = np.zeros((*img.shape, 2, 2))
    h[:, :, 0, 0] = vxx
    h[:, :, 1, 1] = vyy
    h[:, :, 0, 1] = h[:, :, 1, 0] = vxy

    return h


def main():
    parser = argparse.ArgumentParser(
            description="Label raw IPTS touch data and output image files for prototyping")
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

    plt.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(12, 8)

    print("Processing...")
    ims = []
    for i, hm in enumerate(heatmaps):
        elapsed = datetime.datetime.now() - time_start
        print(f"  Frame {i+1}/{len(heatmaps)}, {((i + 1) / len(heatmaps)) * 100:.2f}%, elapsed: {elapsed}")

        hm = scipy.ndimage.gaussian_filter(hm, 1.0)
        hm = np.maximum(hm - np.average(hm), 0.0)

        hs = hessian(hm)
        hs[:, :, 0, 0] = scipy.ndimage.gaussian_filter(hs[:, :, 0, 0], 1.0)
        hs[:, :, 1, 1] = scipy.ndimage.gaussian_filter(hs[:, :, 1, 1], 1.0)
        hs[:, :, 0, 1] = hs[:, :, 1, 0] = scipy.ndimage.gaussian_filter(hs[:, :, 0, 1], 1.0)

        hs_ews, hs_evs = np.linalg.eig(hs)
        ridge = np.sum(np.maximum(hs_ews, 0.0), axis=-1)

        p = list()
        p.append(ax.imshow(ridge.T, animated=True))

        ims.append(p)

    print("Writing images...")
    an = animation.ArtistAnimation(fig, ims, interval=16.66, repeat_delay=1000, blit=True)

    file_out = os.path.abspath(args.file_out[0])
    dir_out = os.path.dirname(file_out)
    os.makedirs(dir_out, exist_ok=True)
    an.save(file_out, writer='imagemagick')

    # rename files
    r = re.compile('(.+)-(\d+).(.+)')
    for file in os.listdir(dir_out):
        m = r.match(file)
        os.rename(f"{dir_out}/{file}", f"{dir_out}/{m[1]}-{int(m[2]):04d}.{m[3]}")


if __name__ == '__main__':
    main()
