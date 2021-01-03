#!/usr/bin/python

from utils.libipts import Parser

import numpy as np
import scipy.ndimage
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

import sys
import os
import re
import datetime
import argparse


# Sobel operator kernels for gradients
SOBEL_X = np.array([
    [1.0, 0.0, -1.0],
    [2.0, 0.0, -2.0],
    [1.0, 0.0, -1.0],
])

SOBEL_Y = SOBEL_X.T


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


def gradient(img):
    vx = scipy.signal.convolve2d(img, SOBEL_X, mode='same')
    vy = scipy.signal.convolve2d(img, SOBEL_Y, mode='same')

    return np.dstack((vx, vy))


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

    plt.axis('off')
    plt.gca().invert_yaxis()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(12, 8)

    print("Processing...")
    ims = []
    for i, hm in enumerate(heatmaps):
        elapsed = datetime.datetime.now() - time_start
        print(f"  Frame {i+1}/{len(heatmaps)}, {((i + 1) / len(heatmaps)) * 100:.2f}%, elapsed: {elapsed}")

        hm = np.maximum(hm - np.average(hm), 0.0)
        hm = scipy.ndimage.gaussian_filter(hm, 1.0)

        gm = np.linalg.norm(gradient(hm), axis=-1)

        p = []
        p.append(ax.imshow(gm.T, vmin=0.0, vmax=1.0, animated=True))

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
