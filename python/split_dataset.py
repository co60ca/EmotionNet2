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

# Designed to be run after preprocessing the inputs into train/
# moves files into directories for each of the classes
import argparse
import os
import random

def main():
    parser = argparse.ArgumentParser(description='Split data')
    parser.add_argument('--split-training', default=0.7, type=float)
    parser.add_argument('--split-validation', default=0.2, type=float)
    parser.add_argument('--split-testing', default=0.1, type=float)
    args = parser.parse_args()
    split(args)

def split(args):
    # eg AM21AFHR.JPG  
    # A series, male...
    traindir = 'train'
    validdir = 'valid'
    testdir = 'test'
    # Find subjects
    available = {}
    for (dirpath, dirnames, filenames) in os.walk(traindir):
        for name in [n for n in filenames \
                     if n.lower().endswith('.jpg')]:
            filen = os.path.join(dirpath, name)
            available[name[0:3]] = True
    # Split subjects
    subjects = list(available.keys())
    random.shuffle(subjects)
    traini = int(len(subjects)*args.split_training)
    validi = int(len(subjects)*args.split_validation)
    testi = int(len(subjects)*args.split_testing)

    # Verify our sets have elements
    if min(traini, validi, testi) < 1:
        raise ValueError('Size of one of the sets is zero')

    test = subjects[:testi]
    valid = subjects[testi:testi+validi]
    train = subjects[testi+validi:]
   

    os.makedirs(testdir, exist_ok=True) 
    os.makedirs(validdir, exist_ok=True) 
    # Move to new directories
    for (dirpath, dirnames, filenames) in os.walk(traindir):
        for filen in [f for f in filenames if f.lower().endswith('.jpg')]:
            path = dirpath.split('/')[1:]
            if filen[0:3] in train:
                pass
                # Should be already there
                #os.renames(os.path.join(dirpath, filen), 
                #          os.path.join(traindir, *dirnames[1:], filen))
            elif filen[0:3] in valid:
                os.renames(os.path.join(dirpath, filen), 
                          os.path.join(validdir, *path, filen))
            elif filen[0:3] in test:
                os.renames(os.path.join(dirpath, filen), 
                          os.path.join(testdir, *path, filen))

if __name__ == '__main__':
    main()
