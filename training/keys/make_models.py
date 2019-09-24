import pumpp
import keras
from keras.layers import concatenate, BatchNormalization, Convolution2D, Bidirectional, GRU, Dense, TimeDistributed

import crema.layers
import crema.feature

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