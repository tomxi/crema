#!/usr/bin/env python
'''CREMA structured beat model: HMM parameters'''

import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm

import crema.utils
from jams.util import smkdirs

OUTPUT_PATH = 'resources'


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(dest='working', type=str,
                        help='Path to working directory')

    parser.add_argument('--pseudo-count', dest='pseudocount',
                        type=float, default=0.5,
                        help='Pseudo-count for self-transitions prior')

    return parser.parse_args(args)


def self_transitions(fname):
    # beats
    #   self-transition
    #   total probability
    # downbeats
    #   self-transition
    #   total probability
    data = crema.utils.load_h5(fname)

    n_total, n_beat_self, n_beat_on = 0, 0, 0

    # we might have multiple annotations per file
    for beat in data['beat/beat']:
        n_total += len(beat) - 1
        n_beat_self += np.sum(beat[1:] == beat[:-1])
        n_beat_on += np.sum(beat > 0)


    n_downbeat_self, n_downbeat_on = 0, 0


    for downbeat in data['beat/downbeat']:
        n_downbeat_self += np.sum(downbeat[1:] == downbeat[:-1])
        n_downbeat_on += np.sum(downbeat > 0)

    return n_total, n_beat_self, n_beat_on, n_downbeat_self, n_downbeat_on


def train(working, pseudocount):
    '''
    Parameters
    ----------
    working : str
        directory that contains the experiment data (h5)
    '''

    # Load the pump
    with open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'rb') as fd:
        pump = pickle.load(fd)

    # Load the training data
    idx_train = pd.read_json('index_train.json')

    n_total, n_beat_self, n_beat_on, n_downbeat_self, n_downbeat_on = 0., 0., 0., 0., 0.

    for track in tqdm(idx_train['id']):
        fname = os.path.join(working, os.path.extsep.join([str(track), 'h5']))
        i_total, i_bs, i_bo, i_ds, i_do = self_transitions(fname)

        n_beat_self += i_bs
        n_beat_on += i_bo
        n_downbeat_self += i_ds
        n_downbeat_on += i_do
        n_total += i_total

    p_beat_self = (n_beat_self + pseudocount) / (n_total + pseudocount)
    p_downbeat_self = (n_downbeat_self + pseudocount) / (n_total + pseudocount)
    p_beat = (n_beat_on + pseudocount) / (n_total + pseudocount)
    p_downbeat = (n_downbeat_on + pseudocount) / (n_total + pseudocount)

    print('# frames = {} + {}'.format(n_total, pseudocount))
    print('# beat self = {} + {}'.format(n_beat_self, pseudocount))
    print('# beat on = {} + {}'.format(n_beat_on, pseudocount))
    print('# downbeat self = {} + {}'.format(n_downbeat_self, pseudocount))
    print('# downbeat on = {} + {}'.format(n_downbeat_on, pseudocount))
    print('P[beat-self] = {:.3g}'.format(p_beat_self))
    print('P[downbeat-self] = {:.3g}'.format(p_downbeat_self))
    print('P[beat] = {:.3g}'.format(p_beat))
    print('P[downbeat] = {:.3g}'.format(p_downbeat))

    pump['beat'].set_transition_beat([p_beat_self, 0])
    pump['beat'].set_transition_down([p_downbeat_self, 0])
    pump['beat'].beat_p_state = p_beat
    pump['beat'].downbeat_p_state = p_downbeat

    with open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'wb') as fd:
        pickle.dump(pump, fd)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    smkdirs(OUTPUT_PATH)

    print('{}: training HMM parameters'.format(__doc__))
    print(params)

    train(params.working, params.pseudocount)
