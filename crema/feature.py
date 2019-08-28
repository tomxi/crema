#!/usr/bin/env python
'''Custom pumpp features'''

import pumpp
from pumpp.feature._utils import to_dtype
from .analyze import get_model


class StructuredChord(pumpp.FeatureExtractor):
    '''Extract Crema style sturcture-chord output as features
    '''

    def __init__(self, name, conv=None, dtype='float32'):

        self.chord_model = get_model('chord')

        sr = self.chord_model.pump['chord_struct'].sr
        hop_length = self.chord_model.pump['chord_struct'].hop_length

        super(StructuredChord, self).__init__(name, sr, hop_length,
                                              conv=conv, dtype=dtype)

        self.register('bass', 13, self.dtype)
        self.register('pitch', 12, self.dtype)
        self.register('root', 13, self.dtype)

    def transform_audio(self, y):
        '''Run audio through CREMA chord model

        Parameters
        ----------
        y : np.ndarray
            Audio buffer

        Returns
        -------
        data : dict
            data['bass'] = np.ndarray, shape=(n_frames, 13)
            data['root'] = np.ndarray, shape=(n_frames, 13)
            data['pitch'] = np.ndarray, shape=(n_frames, 12)

        '''

        outputs = self.chord_model.outputs(y=y, sr=self.sr)

        return {'bass': to_dtype(outputs['chord_bass'][self.idx], self.dtype),
                'root': to_dtype(outputs['chord_root'][self.idx], self.dtype),
                'pitch': to_dtype(outputs['chord_pitch'][self.idx], self.dtype)
                }
