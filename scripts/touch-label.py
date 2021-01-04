#!/usr/bin/python

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

from libipts import Parser

import numpy as np
import scipy.ndimage
from skimage import measure

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

import re
import itertools
import datetime
import argparse


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


def get_bbox(data, label):
    d = data == label
    rows = np.any(d, axis=1)
    cols = np.any(d, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def main():
    delta = 0.075

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
        blobs = hm > delta
        labels = measure.label(blobs, background=0)

        maximas = get_local_maximas(hm, delta)

        counts = np.zeros((np.max(labels) + 1,))
        for mu in maximas:
            counts[labels[mu[0], mu[1]]] += 1

        p = list()
        p.append(ax.imshow(hm.T, cmap='gray', animated=True))
        p.append(ax.imshow(labels.T, cmap='viridis', alpha=0.5, animated=True))

        for mu in maximas:
            color = 'black' if counts[labels[mu[0], mu[1]]] > 1 else 'red'
            p += ax.plot(mu[0], mu[1], 'b+', ms=10, color=color, animated=True)

        nf, no = 0, 0
        for l in range(1, np.max(labels) + 1):
            if counts[l] == 1:
                color = 'red'
                text = f'F{nf}'
                nf += 1
            else:
                color = 'lightgray'
                text = f'O{no}'
                no += 1

            x1, x2, y1, y2 = get_bbox(labels, l)
            r = patches.Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1 + 1.0, y2 - y1 + 1.0, fill=False, linewidth=1, color=color)
            ax.add_artist(r)
            p.append(r)

            p.append(ax.text(x1 - 0.5, y1 - 0.5, f"{text} ", fontsize='small', horizontalalignment='right', verticalalignment='top', color=color))

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
