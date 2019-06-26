#!/usr/bin/env python
'''CREMA beats and downbeats'''

import argparse
import os
import sys
from glob import glob
import pickle

import pandas as pd
import keras as K

from sklearn.model_selection import ShuffleSplit

import pescador
import pumpp
import librosa
import crema.utils
import crema.layers
from jams.util import smkdirs

OUTPUT_PATH = 'resources'


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--max_samples', dest='max_samples', type=int,
                        default=128,
                        help='Maximum number of samples to draw per streamer')

    parser.add_argument('--patch-duration', dest='duration', type=float,
                        default=16.0,
                        help='Duration (in seconds) of training patches')

    parser.add_argument('--seed', dest='seed', type=int,
                        default='20190625',
                        help='Seed for the random number generator')

    parser.add_argument('--train-streamers', dest='train_streamers', type=int,
                        default=1024,
                        help='Number of active training streamers')

    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=32,
                        help='Size of training batches')

    parser.add_argument('--rate', dest='rate', type=int,
                        default=8,
                        help='Rate of pescador stream deactivation')

    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=1000,
                        help='Maximum number of epochs to train for')

    parser.add_argument('--epoch-size', dest='epoch_size', type=int,
                        default=2048,
                        help='Number of batches per epoch')

    parser.add_argument('--early-stopping', dest='early_stopping', type=int,
                        default=100,
                        help='# epochs without improvement to stop')

    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int,
                        default=10,
                        help='# epochs before reducing learning rate')

    parser.add_argument(dest='working', type=str,
                        help='Path to working directory')

    return parser.parse_args(args)


def make_sampler(max_samples, duration, pump, seed):

    n_frames = librosa.time_to_frames(duration,
                                      sr=pump['mel'].sr,
                                      hop_length=pump['mel'].hop_length)

    return pump.sampler(max_samples, n_frames, random_state=seed)


def val_sampler(max_duration, pump, seed):
    '''validation sampler'''
    n_frames = librosa.time_to_frames(max_duration,
                                      sr=pump['mel'].sr,
                                      hop_length=pump['mel'].hop_length)

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


def construct_model(pump):

    model_inputs = ['mel/mag']

    # Build the input layer
    layers = pump.layers()

    x_mag = layers['mel/mag']

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x_mag)

    # First convolutional filter: a single 7x7
    conv1 = K.layers.Convolution2D(32, (7, 7),
                                   padding='same',
                                   activation='relu',
                                   data_format='channels_last')(x_bn)

    # Second convolutional filter: a bank of full-height filters
    conv2 = K.layers.Convolution2D(64, (1, int(conv1.shape[2])),
                                   padding='valid', activation='relu',
                                   data_format='channels_last')(conv1)

    squeeze_c = crema.layers.SqueezeLayer(axis=-1)(x_bn)
    squeeze = crema.layers.SqueezeLayer(axis=2)(conv2)

    rnn_in = K.layers.concatenate([squeeze, squeeze_c])
    # BRNN layer
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(rnn_in)
    r1bn = K.layers.BatchNormalization()(rnn1)
    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(r1bn)

    r2bn = K.layers.BatchNormalization()(rnn2)

    # Skip connections to the convolutional layers
    codec = K.layers.concatenate([r2bn, r1bn, squeeze])
    codecbn = K.layers.BatchNormalization()(codec)

    p0 = K.layers.Dense(1, activation='sigmoid')
    p1 = K.layers.Dense(1, activation='sigmoid')

    beat = K.layers.TimeDistributed(p0, name='beat')(codec)
    downbeat = K.layers.TimeDistributed(p1, name='downbeat')(codec)

    model = K.models.Model([x_mag],
                           [beat, downbeat])

    model_outputs = ['beat/beat', 'beat/downbeat']

    return model, model_inputs, model_outputs


def train(working, max_samples, duration, rate,
          batch_size, epochs, epoch_size, 
          early_stopping, reduce_lr, seed):
    '''
    Parameters
    ----------
    working : str
        directory that contains the experiment data (h5)

    max_samples : int
        Maximum number of samples per streamer

    duration : float
        Duration of training patches

    batch_size : int
        Size of batches

    rate : int
        Poisson rate for pescador

    epochs : int
        Maximum number of epoch

    epoch_size : int
        Number of batches per epoch

    early_stopping : int
        Number of epochs before early stopping

    reduce_lr : int
        Number of epochs before reducing learning rate

    seed : int
        Random seed
    '''

    # Load the pump
    with open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'rb') as fd:
        pump = pickle.load(fd)

    # Build the sampler
    sampler = make_sampler(max_samples, duration, pump, seed)

    sampler_val = val_sampler(10 * 60, pump, seed)

    # Build the model
    model, inputs, outputs = construct_model(pump)

    # Load the training data
    idx_train_ = pd.read_json('index_train.json')

    # Split the training data into train and validation
    splitter_tv = ShuffleSplit(n_splits=1, test_size=0.25,
                               random_state=seed)
    train, val = next(splitter_tv.split(idx_train_))

    idx_train = idx_train_.iloc[train]
    idx_val = idx_train_.iloc[val]

    gen_train = data_generator(working,
                               idx_train['id'].values, sampler, epoch_size,
                               augment=True,
                               rate=rate,
                               mode='with_replacement',
                               random_state=seed)

    gen_train = pescador.maps.keras_tuples(pescador.maps.buffer_stream(gen_train(), 
                                                                       batch_size,
                                                                       axis=0),
                                           inputs=inputs,
                                           outputs=outputs)

    gen_val = val_generator(working, idx_val['id'].values, sampler_val,
                            augment=True)

    validation_size = gen_val.n_streams

    gen_val = pescador.maps.keras_tuples(gen_val(), inputs=inputs, outputs=outputs)

    loss = {'beat': 'binary_crossentropy',
            'downbeat': 'binary_crossentropy'}

    metrics = {'beat': 'accuracy', 'downbeat': 'accuracy'}

    monitor = 'val_loss'

    sgd = K.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(sgd, loss=loss, metrics=metrics)

    # Store the model
    model_spec = K.utils.serialize_keras_object(model)
    with open(os.path.join(OUTPUT_PATH, 'model_spec.pkl'), 'wb') as fd:
        pickle.dump(model_spec, fd)

    # Construct the weight path
    weight_path = os.path.join(OUTPUT_PATH, 'model.h5')

    # Build the callbacks
    cb = []
    cb.append(K.callbacks.ModelCheckpoint(weight_path,
                                          save_best_only=True,
                                          verbose=1,
                                          monitor=monitor))

    cb.append(K.callbacks.ReduceLROnPlateau(patience=reduce_lr,
                                            verbose=1,
                                            monitor=monitor))

    cb.append(K.callbacks.EarlyStopping(patience=early_stopping,
                                        verbose=1,
                                        monitor=monitor))

    # Fit the model
    model.fit_generator(gen_train, epoch_size, epochs,
                        validation_data=gen_val,
                        validation_steps=validation_size,
                        callbacks=cb)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    smkdirs(OUTPUT_PATH)

    version = crema.utils.increment_version(os.path.join(OUTPUT_PATH,
                                                         'version.txt'))

    print('{}: training'.format(__doc__))
    print('Model version: {}'.format(version))
    print(params)

    train(params.working,
          params.max_samples, params.duration,
          params.rate,
          params.batch_size,
          params.epochs, params.epoch_size,
          params.early_stopping,
          params.reduce_lr,
          params.seed)
