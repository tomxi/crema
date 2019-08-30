#!/usr/bin/env python
'''Utilities for training the key model'''

import crema.feature
import pumpp
import pandas as pd

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
    
    # don't save the pump, cuz you can't de-serialize them with custom keras layers :(
    # with open(os.path.join(OUTPUT_PATH, 'pump.pkl'), 'wb') as fd:
    #     pickle.dump(pump, fd)
    
    return pump


def get_ann_audio_guitarset(data_home, gs_index_json):
    '''Get a list of annotations and audio files from a gs_index_json.

    Parameters
    ----------
    gs_index_json : str
        The json file to pull data
    Returns
    -------
    pairs : list of tuples (audio_file, annotation_file)
    '''
    
    audio_paths = []
    anno_paths = []
    
    index_df = pd.read_json(gs_index_json)
    for track_id in index_df:
        audio_rel_path, audio_md5 = index_df[track_id].audio_mic
        jams_rel_path, jams_md5 = index_df[track_id].jams
        
        aud_path = os.path.join(data_home, audio_rel_path)
        jams_path = os.path.join(data_home, jams_rel_path)
        assert os.path.isfile(aud_path) and os.path.isfile(jams_path)
        
        audio_paths.append(aud_path)
        anno_paths.append(jams_path)

    paired = list(zip(audio_paths, anno_paths))

    return paired
