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


def make_pump():
    p_feature = crema.feature.StructuredChord(name='chord_struct',
                                              conv='tf')
    
    p_key_tag = pumpp.task.KeyTagTransformer(name='key_tag',
                                             sr=p_feature.sr,
                                             hop_length=p_feature.hop_length,
                                             sparse=True)
    
    p_key_struct = pumpp.task.KeyTransformer(name='key_struct',
                                             sr=p_feature.sr,
                                             hop_length=p_feature.hop_length,
                                             sparse=True)
    
    pump = pumpp.Pump(p_feature, p_key_tag, p_key_struct)
    
    # TODO save the pump
    with open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'wb') as fd:
        pickle.dump(pump, fd)
    
    return pump



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

    stream = tqdm(crema.utils.get_ann_audio_json(params.data_home, params.index_json),
                  desc='Converting training data')
    Parallel(n_jobs=n_jobs)(delayed(convert)(aud, ann,
                                             pump,
                                             params.output_path)
                                   for aud, ann in stream)

    if params.augment_path:
        stream = tqdm(crema.utils.get_ann_audio_json(params.data_home, params.index_json),
                      desc='Converting augmented data')
        Parallel(n_jobs=n_jobs)(delayed(convert)(aud, ann,
                                                 pump,
                                                 params.output_path)
                                       for aud, ann in stream)
