#!/usr/bin/python

from utils.libipts import Parser

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

import os, re
import itertools
import datetime
import argparse


VTH_TOUCH = 10.0 / 255.0


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


class Contact:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0

        self.ev1 = 0.0
        self.ev2 = 0.0

        self.qx1 = 0.0
        self.qy1 = 0.0
        self.qx2 = 0.0
        self.qy2 = 0.0

        self.w_max = 0.0
        self.angle = 0.0


class Cluster:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.xx = 0.0
        self.yy = 0.0
        self.xy = 0.0
        self.w = 0.0
        self.w_max = 0.0

    def extend(self, xy, heatmap, visited):
        x, y = xy

        if x < 0 or y < 0 or x >= heatmap.shape[0] or y >= heatmap.shape[1]:
            return

        if heatmap[x, y] < VTH_TOUCH:
            return

        if visited[x, y]:
            return

        visited[x, y] = True

        # add to cluster
        w = heatmap[x, y]
        self.w += w
        self.w_max = w if w > self.w_max else self.w_max
        self.x += w * x
        self.y += w * y
        self.xx += w * x * x
        self.yy += w * y * y
        self.xy += w * x * y

        # continue recursively
        self.extend((x - 1, y), heatmap, visited)
        self.extend((x + 1, y), heatmap, visited)
        self.extend((x, y - 1), heatmap, visited)
        self.extend((x, y + 2), heatmap, visited)

    def to_contact(self):
        c = Contact()

        c.x = self.x / self.w
        c.y = self.y / self.w

        vx = (self.xx - (self.x * self.x / self.w)) / self.w;
        vy = (self.yy - (self.y * self.y / self.w)) / self.w;
        cv = (self.xy - (self.x * self.y / self.w)) / self.w;
        sqrtd = np.sqrt((vx - vy)**2 + 4.0 * cv**2)

        c.ev1 = (vx + vy + sqrtd) / 2.0
        c.ev2 = (vx + vy - sqrtd) / 2.0

        c.qx1 = vx + cv - c.ev2
        c.qy1 = vy + cv - c.ev2
        c.qx2 = vx + cv - c.ev1
        c.qy2 = vy + cv - c.ev1

        d1 = np.hypot(c.qx1, c.qy1);
        d2 = np.hypot(c.qx2, c.qy2);

        if d1 == 0:
            print("Error: d1 == 0")
            d1 = 1.0

        if d2 == 0:
            print("Error: d2 == 0")
            d2 = 1.0

        c.qx1 /= d1;
        c.qy1 /= d1;
        c.qx2 /= d2;
        c.qy2 /= d2;

        c.w_max = self.w_max;

        c.angle = np.pi - np.arctan2(c.qx1, c.qy1);
        if c.angle < 0:
            c.angle += np.pi
        if c.angle > np.pi:
            c.angle -= np.pi

        return c


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

        visited = np.full(hm.shape, False, dtype=np.bool)
        contacts = []

        for x1, x2 in itertools.product(range(hm.shape[0]), range(hm.shape[1])):
            if hm[x1, x2] < VTH_TOUCH:
                continue

            if visited[x1, x2]:
                continue

            cluster = Cluster()
            cluster.extend((x1, x2), hm, visited)

            contact = cluster.to_contact()
            if contact is not None and contact.ev2 > 0:
                contacts.append(contact)

        p = []
        p.append(ax.imshow(hm.T, vmin = 0.0, vmax = 1.0, animated=True))

        for c in contacts:
            p += ax.plot(c.x, c.y, 'b+', ms=10, color='red')

            width, height = 2.0 * nstd * np.sqrt((c.ev2, c.ev1))
            e = Ellipse(xy=(c.x, c.y), width=width, height=height, angle=np.degrees(c.angle), facecolor='none', edgecolor='red')
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
