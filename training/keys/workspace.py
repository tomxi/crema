import os
from glob import glob

import librosa
import pumpp
import pescador
import keras
from keras.layers import concatenate, BatchNormalization, Convolution2D, Bidirectional, GRU, Dense, TimeDistributed

import crema.utils
import crema.layers

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


def construct_model(pump):
    model_inputs = ['chord_struct/bass', 'chord_struct/pitch']
    model_outputs = ['key_tag/tag', 'key_struct/tonic', 'key_struct/pitch_profile']
    
    x_dict = pump.layers()
    x_list = [x_dict[k] for k in model_inputs]
    x = concatenate(x_list, axis=2, name='input')
    
    x_bn = BatchNormalization()(x)

    # First convolutional filter: a single 5x5
    conv1 = Convolution2D(1, (5, 5),
                          padding='same',
                          activation='relu',
                          data_format='channels_last')(x_bn)

    c1bn = BatchNormalization()(conv1)

    # Second convolutional filter: a bank of full-height filters
    conv2 = Convolution2D(12*6, (1, int(conv1.shape[2])),
                                   padding='valid', activation='relu',
                                   data_format='channels_last')(c1bn)

    c2bn = BatchNormalization()(conv2)
    
    # Squeeze out the frequency dimension
    squeeze = crema.layers.SqueezeLayer(axis=2)(c2bn)

    # BRNN layer
    rnn1 = Bidirectional(GRU(128, return_sequences=True))(squeeze)

    r1bn = BatchNormalization()(rnn1)

    rnn2 = Bidirectional(GRU(128, return_sequences=True))(r1bn)
    
    # 1: pitch profile predictor
    profile = Dense(pump.fields['key_struct/pitch_profile'].shape[1],
                    activation='sigmoid')

    profile_p =  TimeDistributed(profile, name='key_profile')(rnn2)

    # 2: tonic predictor
    tonic = Dense(13, activation='softmax')
    tonic_p = TimeDistributed(tonic, name='key_tonic')(rnn2)


    # 3: merge layer
    codec = concatenate([rnn2, profile_p, tonic_p], name='codec')

    codecbn = BatchNormalization()(codec)

    p0 = Dense(len(pump['key_tag'].vocabulary()),
                activation='softmax',
                bias_regularizer=keras.regularizers.l2())

    tag = TimeDistributed(p0, name='key_tag')(codecbn)
    
    model = keras.models.Model(x_list, [tag, tonic_p, profile_p])
    
    return model, model_inputs, model_outputs