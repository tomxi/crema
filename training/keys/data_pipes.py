import os
from glob import glob

import librosa
import pumpp
import pescador
import numpy as np

import crema.utils

class TransSampler(pumpp.sampler.Sampler):
    def __init__(self, n_samples, duration, *ops, **kwargs):
        super(TransSampler, self).__init__(n_samples, duration, *ops, **kwargs)

        self.tag_encoder = pumpp.task.KeyTagTransformer(sparse=True).encoder
    
    def sample(self, data, interval):
        '''Sample a patch from the data object

        Parameters
        ----------
        data : dict
            A data dict as produced by pumpp.Pump.transform

        interval : slice
            The time interval to sample

        Returns
        -------
        data_slice : dict
            `data` restricted to `interval`.
        '''
        data_slice = super(TransSampler, self).sample(data, interval)
        
        transpose_amount = self.rng.randint(0, 12)
        
        for key in data_slice:
            if 'chord' in key:
                if data_slice[key].shape[2] == 13:
                    to_transpose = data_slice[key][:, :, :-1, :]
                    data_slice[key][:, :, :-1, :] = np.roll(to_transpose, transpose_amount, 2)
                elif data_slice[key].shape[2] == 12:
                    data_slice[key] = np.roll(data_slice[key], transpose_amount, 2)
                else:
                    raise IndexError            
        
        key_tags = data_slice['key_tag/tag']
        tonics = [keytag.split(':')[0] for keytag in self.tag_encoder.inverse_transform(key_tags.squeeze())]
        modes = [keytag.split(':')[1] for keytag in self.tag_encoder.inverse_transform(key_tags.squeeze())]
        new_tonics = librosa.midi_to_note(librosa.note_to_midi(tonics) + transpose_amount, octave=False)

        for i, new_tonic in enumerate(new_tonics):
            new_key = ':'.join([new_tonic, modes[i]])
            data_slice['key_tag/tag'][0, i, :] = self.tag_encoder.transform([new_key])
            profile, tonic = pumpp.task.key._encode_key_str(new_key, True)
            data_slice['key_struct/pitch_profile'][0, i, :] = profile
            data_slice['key_struct/tonic'][0, i, :] = tonic
        
        return data_slice


def train_sampler(max_samples, duration, pump, seed):
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


def aug_train_sampler(max_samples, duration, pump, seed):
    '''stochastic trainning sampler 
    with data augmentation done by random transposition
    '''
    n_frames = librosa.time_to_frames(duration, 
                                      sr=pump['chord_struct'].sr,
                                      hop_length=pump['chord_struct'].hop_length)

    return TransSampler(max_samples, n_frames, random_state=seed, *pump.ops)

    
def data_sampler(fname, sampler):
    '''Generate samples from a specified h5 file'''
    for datum in sampler(crema.utils.load_h5(fname)):
        yield datum

def data_sampler_fl(fname):
    while True:
        yield crema.utils.load_h5(fname)

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


def val_generator(working, tracks, augment=True):
    '''validation generator, deterministic roundrobin'''
    seeds = []
    for track in tracks:
        fname = os.path.join(working,
                             os.path.extsep.join([track, 'h5']))
        seeds.append(pescador.Streamer(data_sampler_fl, fname))

        if augment:
            for fname in sorted(glob(os.path.join(working,
                                                  '{}.*.h5'.format(track)))):
                seeds.append(pescador.Streamer(data_sampler_fl, fname))

    # Send it all to a mux
    return pescador.RoundRobinMux(seeds)


