#!/usr/bin/env python
'''CREMA Chord model'''

import numpy as np
import mir_eval
import pumpp

from .base import CremaModel
from ..feature import StructuredChord


class KeyModel(CremaModel):

    def __init__(self):
        self._instantiate('key', make_pump_lambda=self.make_pump_)

    def predict(self, filename=None, y=None, sr=None, outputs=None):
        '''Key prediction

        Parameters
        ----------
        filename : str
            Path to the audio file to analyze

        y, sr : np.ndarray, number>0

            Audio signal in memory to analyze

        outputs : dict `{str: np.ndarray}`

            Pre-computed model outputs, as given by ``KeyModel.outputs``.

        .. note:: At least one of `filename`, `y, sr`, or `outputs`
            must be provided.

        Returns
        -------
        jams.Annotation, namespace='key_mode'

            The chord estimate for the given signal.

        Examples
        --------
        >>> import crema
        >>> import librosa
        >>> model = crema.models.key.KeyModel()
        >>> key_est = model.predict(filename=librosa.util.example_audio_file())
        >>> key_est
        TODO
        '''
        if outputs is None:
            outputs = self.outputs(filename=filename, y=y, sr=sr)

        output_key = self.model.output_names[0]
        pump_op = self.pump[output_key]

        ann = super(KeyModel, self).predict(y=y, sr=sr, filename=filename,
                                            outputs=outputs)

        return ann

    def make_pump_(self):
        p_feature = StructuredChord(name='chord_struct',
                                    conv='tf')
    
        p_key_tag = pumpp.task.KeyTagTransformer(name='key_tag',
                                                 sr=p_feature.sr,
                                                 hop_length=p_feature.hop_length,
                                                 sparse=True)
        
        p_key_struct = pumpp.task.KeyTransformer(name='key_struct',
                                                 sr=p_feature.sr,
                                                 hop_length=p_feature.hop_length,
                                                 sparse=True)
        
        return pumpp.Pump(p_feature, p_key_tag, p_key_struct)
