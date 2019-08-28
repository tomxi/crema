#!/usr/bin/env python
'''CREMA analyzer interface'''

import argparse
import sys

import librosa
import jams

from . import models

__MODELS__ = {}

__all__ = ['analyze', 'get_model', 'main']


def analyze(filename=None, y=None, sr=None):
    '''Analyze a recording for all tasks.

    Parameters
    ----------
    filename : str, optional
        Path to audio file

    y : np.ndarray, optional
    sr : number > 0, optional
        Audio buffer and sampling rate

    .. note:: At least one of `filename` or `y, sr` must be provided.

    Returns
    -------
    jam : jams.JAMS
        a JAMS object containing all estimated annotations

    Examples
    --------
    >>> from crema.analyze import analyze
    >>> import librosa
    >>> jam = analyze(filename=librosa.util.example_audio_file())
    >>> jam
    <JAMS(file_metadata=<FileMetadata(...)>,
          annotations=[1 annotation],
          sandbox=<Sandbox(...)>)>
    >>> # Get the chord estimates
    >>> chords = jam.annotations['chord', 0]
    >>> chords.to_dataframe().head(5)
           time  duration  value  confidence
    0  0.000000  0.092880  E:maj    0.336977
    1  0.092880  0.464399    E:7    0.324255
    2  0.557279  1.021678  E:min    0.448759
    3  1.578957  2.693515  E:maj    0.501462
    4  4.272472  1.486077  E:min    0.287264
    '''

    _load_models()

    jam = jams.JAMS()
    # populate file metadata

    jam.file_metadata.duration = librosa.get_duration(y=y, sr=sr,
                                                      filename=filename)

    for namespace, model in __MODELS__.items():
        jam.annotations.append(model.predict(filename=filename, y=y, sr=sr))

    return jam


def get_model(namespace):
    '''Get a specific model.

    Parameters
    ----------
    namespace : str
        one of ['chord', 'beat', 'key']

    Returns
    -------
    model : crema.models.<task>.<Task>Model or None
        access the __MODELS__ dictionary. If namespace is not in __MODELS__
        return None

    '''

    _load_models()

    try:
        return __MODELS__[namespace]
    except KeyError:
        print('No model for the "{}" task is loaded!'.format(namespace))
        return None

def parse_args(args):  # pragma: no cover

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('filename',
                        type=str,
                        help='Audio file to process')

    parser.add_argument('-o', '--output', dest='output',
                        type=argparse.FileType('w'),
                        default=sys.stdout,
                        help='Path to store JAMS output')

    return parser.parse_args(args)


def main():  # pragma: no cover
    params = parse_args(sys.argv[1:])
    jam = analyze(params.filename)
    jam.save(params.output)


# Populate models array
def _load_models():
    '''This helper builds and caches the model objects.'''
    global __MODELS__

    if not __MODELS__:
        __MODELS__['chord'] = models.chord.ChordModel()


if __name__ == '__main__':  # pragma: no cover
    main()
