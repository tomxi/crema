Structured key estimation
===========================

This model is based on the chord model in crema


Model architecture
------------------
TO CHANGE
The model accepts as input a harmonic constant-Q spectrogram:

- `HCQT(name='cqt', sr=44100, hop_length=4096, n_octaves=6,
        harmonics=[1, 2], over_sample=3, log=True, conv='channels_last')`

The model has 9 layers (not counting reshaping/merging and batchnorm) defined as follows:

- `Input`
- `BatchNorm`
- `Conv2D(1, (5, 5), activation='relu', padding='same')`
- `Conv2D(72, (1, 216), activation='relu', padding='valid')`
- `Squeeze`
- `Bidirectional(GRU(128))`
- `Bidirectional(GRU(128))`  <-- the encoding layer
- `Root, Pitches, Bass`      <-- the structured output
- `Chord`                    <-- the decoded chord label output

The output chord vocabulary is pumpp's `3567s` set (170).

Training
--------
TO CHANGE
We use MUDA for data augmentation: +- 6 semitones for each training track.

Batches are managed by pescador with the following parameters:

- 8-second patches
- 128 samples per active streamer
- rate=16
- 1024 active streamers
- 16 patches per training batch
- 1024 batches per epoch
- 1024 batches per validation
- learning rate reduction at 10
- early stopping at 100
- max 500 epochs


Sequential modeling
-------------------
TO CHANGE
Finally, the decoder model uses discriminative viterbi decoding over chord tag
labels.  The HMM parameters are estimated from the training set statistics.
