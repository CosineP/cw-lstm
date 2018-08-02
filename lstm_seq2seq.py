#!/usr/bin/env python

'''Sequence to sequence example in Keras (character-level).

This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

# Summary of the algorithm

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.

# Data download

English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip

Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/

# References

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
'''
from __future__ import print_function

# Start with the argument parsing, before imports, because importing takes a
# long time and we should know immediately if we gave malformed arguments
import argparse

parser = argparse.ArgumentParser(description="train NN to generate CWs")
parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of times to iterate through the entire training set.")
args = parser.parse_args()

# from keras.models import Model
# from keras.layers import Input, LSTM, Dense
import model as lstm_model
import data

batch_size = 64  # Batch size for training.
epochs = args.epochs
# num_samples = 2000  # Number of samples to train on.
num_samples = 1000000000  # Number of samples to train on.
# data headings are [content, cw]

encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens = data.load(num_samples)

model, encoder_model, decoder_model = lstm_model.generate_models(num_encoder_tokens, num_decoder_tokens)

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

for seq_index in range(50):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = data.decode_sequence(encoder_model, decoder_model, input_seq)
    print('-')
    print('Toot:', input_texts[seq_index])
    print('CW:', decoded_sentence)

