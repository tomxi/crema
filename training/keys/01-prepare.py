#!/usr/bin/env python
'''CREMA structured key model'''

import argparse
import sys
import os
import pickle

from tqdm import tqdm
from joblib import Parallel, delayed

from jams.util import smkdirs

import pumpp

import crema.utils
from train_utils import make_pump, get_ann_audio_guitarset

OUTPUT_PATH = 'resources'


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--augmentation-path', dest='augment_path', type=str,
                        default=None,
                        help='Path for augmented data (optional)')

    parser.add_argument('data_home', type=str,
                        help='Path for directory containing the Dataset')

    parser.add_argument('index_json', type=str,
                        help='json file path containing pairing info')

    parser.add_argument('output_path', type=str,
                        help='Path to store pump output')

    return parser.parse_args(args)


def convert(aud, jam, pump, outdir):
    data = pump.transform(aud, jam)
    fname = os.path.extsep.join([os.path.join(outdir, crema.utils.base(aud)),
                                'h5'])
    crema.utils.save_h5(fname, **data)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    smkdirs(OUTPUT_PATH)
    smkdirs(params.output_path)

    print('{}: pre-processing'.format(__doc__))
    print(params)
    pump = make_pump()
    n_jobs = 1 # can't parallelize due to custom keras layers

    stream = tqdm(get_ann_audio_guitarset(params.data_home, params.index_json),
                  desc='Converting training data')
    Parallel(n_jobs=n_jobs)(delayed(convert)(aud, ann,
                                             pump,
                                             params.output_path)
                            for aud, ann in stream)

    if params.augment_path:
        stream = tqdm(get_ann_audio_guitarset(params.data_home, params.index_json),
                      desc='Converting augmented data')
        Parallel(n_jobs=n_jobs)(delayed(convert)(aud, ann,
                                                 pump,
                                                 params.output_path)
                                for aud, ann in stream)
