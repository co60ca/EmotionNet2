# This file is part of EmotionNet2 a system for predicting facial emotions
# Copyright (C) 2018  Brad Kennedy
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

# Edited from dlib example
import sys
import os

import dlib
from skimage import io
import argparse
import math
import numpy as np
import skimage.transform

def transform(args, files):

    detector = dlib.get_frontal_face_detector()
    if args.window:
        win = dlib.image_window()
    progress = 1
    count = len(files)
    for line in files:
        print("Processing file: {} {}/{}".format(line, progress, count))
        progress += 1
        img = io.imread(line)
        dets, scores, idx = detector.run(img, 1, args.threshold)
        print("Number of faces detected: {}".format(len(dets)))
        if args.ignore_multi and len(dets) > 1:
            print("Skipping image with more then one face")
            continue
            
        if len(dets) == 0:
            print('Skipping image as no faces found')
            continue

        d = dets[0]

        (ymax, xmax, _) = img.shape
        g = args.grow
        l, t, r, b = max(d.left()-g, 0), max(d.top()-g, 0), \
                     min(d.right()+g, xmax), min(d.bottom()+g, ymax)
        # Proportion check
        if ((r-l)*(b-t))/(xmax * ymax) < args.min_proportion:
            print('Image proportion too small, skipping')

        if args.window:
            win.clear_overlay()
            win.set_image(img)
            win.add_overlay(dets)
            dlib.hit_enter_to_continue()
        
        img = img[np.arange(t, b),:,:]
        img = img[:, np.arange(l, r), :]
        if args.resize:
            img = skimage.transform.resize(img, 
                  (args.row_resize, args.col_resize))
        io.imsave(args.o + '/' + os.path.basename(line), img)

def main():
    parser = argparse.ArgumentParser(description="Preprocesses photos to" +
        " their face detected version")
    # Input/Output
    parser.add_argument('-o')
    # Lower threshold is more lossy
    parser.add_argument('--threshold', default=0.0, type=float)
    parser.add_argument('--window', default=False, type=bool)
    parser.add_argument('--ignore-multi', default=True, type=bool)
    parser.add_argument('--grow', default=10, type=int)
    parser.add_argument('--resize', default=True, type=bool)
    parser.add_argument('--row-resize', default=512, type=int)
    parser.add_argument('--col-resize', default=512, type=int)
    parser.add_argument('--min-proportion', default=0.1, type=float)

    args = parser.parse_args()

    files=[line.strip() for line in sys.stdin]

    transform(args, files)
if __name__ == '__main__':
    main()
