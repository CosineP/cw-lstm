#!/usr/bin/env python

# Start with the argument parsing, before imports, because importing takes a
# long time and we should know immediately if we gave malformed arguments
import argparse

parser = argparse.ArgumentParser(description="train NN to generate CWs")
parser.add_argument("--weights", "-w", type=str, default='s2s.h5', help="The weights file (.h5)")
args = parser.parse_args()

import model as lstm_model
import data

encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens = data.load()

_, encoder_model, decoder_model = lstm_model.generate_models(num_encoder_tokens, num_decoder_tokens)
# Load model
encoder_model.load_weights(args.weights)

for seq_index in range(actual_num_samples * 0.8, actual_num_samples * 0.8 + 50):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = data.decode_sequence(encoder_model, decoder_model, input_seq)
    print('-')
    print('Toot:', input_texts[seq_index])
    print('CW:', decoded_sentence)

