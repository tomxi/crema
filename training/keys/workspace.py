import os

import librosa
import pumpp
import pescador

import crema.utils

def make_sampler(max_samples, duration, pump, seed):
    '''stochastic training sampler'''
    n_frames = librosa.time_to_frames(duration, 
                                      sr=pump['chord_struct'].sr,
                                      hop_length=pump['chord_struct'].hop_length)

    return pump.sampler(max_samples, n_frames, random_state=seed)


def val_sampler(max_duration, pump, seed):
    '''validation sampler'''
    n_frames = librosa.time_to_frames(max_duration,
                                      sr=pump['chord_struct'].sr,
                                      hop_length=pump['chord_struct'].hop_length)

    return pumpp.sampler.VariableLengthSampler(None, 32, n_frames,
                                               *pump.ops,
                                               random_state=seed)


def data_sampler(fname, sampler):
    '''Generate samples from a specified h5 file'''
    for datum in sampler(crema.utils.load_h5(fname)):
        yield datum
        

def data_generator(working, tracks, sampler, k, augment=True, rate=8, **kwargs):
    '''Generate a data stream from a collection of tracks and a sampler'''

    seeds = []

    for track in tracks:
        fname = os.path.join(working,
                             os.path.extsep.join([track, 'h5']))
        seeds.append(pescador.Streamer(data_sampler, fname, sampler))

        if augment:
            for fname in sorted(glob(os.path.join(working,
                                                  '{}.*.h5'.format(track)))):
                seeds.append(pescador.Streamer(data_sampler, fname, sampler))

    # Send it all to a mux
    return pescador.StochasticMux(seeds, k, rate, **kwargs)


def val_generator(working, tracks, sampler, augment=True):
    '''validation generator, deterministic roundrobin'''
    seeds = []
    for track in tracks:
        fname = os.path.join(working,
                             os.path.extsep.join([track, 'h5']))
        seeds.append(pescador.Streamer(data_sampler, fname, sampler))

        if augment:
            for fname in sorted(glob(os.path.join(working,
                                                  '{}.*.h5'.format(track)))):
                seeds.append(pescador.Streamer(data_sampler, fname, sampler))

    # Send it all to a mux
    return pescador.RoundRobinMux(seeds)
