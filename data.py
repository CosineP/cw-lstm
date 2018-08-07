#!/usr/bin/env python

import csv
import json
import numpy as np

data_path = 'data/toots.csv'
# data headings are [content, cw]
max_decoder_seq_length = None

# For consistency + convenience, WE DECIDE what characters are valid
# But don't worry, I examined the toots and this is a pretty damn good set
def get_characters():
    with open("data/symbols.txt") as s:
        # Add the control characters as well
        return set([c[0] for c in s.readlines()]) | set(['\x05', '\x02', '\x03'])

def load(num_samples=-1):
    # I give up! Passing things around is a mess! I'm using a global! Shoot me!
    global max_decoder_seq_length
    # Vectorize the data.
    input_texts = []
    target_texts = []
    characters = get_characters()
    with open(data_path, 'r', encoding='utf-8') as f:
        tootreader = csv.reader(f)
        raw_toots = list(tootreader)
    actual_num_samples = min(num_samples, len(raw_toots) - 1) if num_samples != -1 else len(raw_toots) - 1
    num_with_cw = 0.
    for toot in raw_toots[:actual_num_samples]:
        # Most toots limited to 500 so limiting to 500 doesn't kill lots of data
        # but does make it a lot easer on the GPU
        input_text = toot[0][:500] 
        # CWs can be really long in jokes but that's not really what we care
        # about. 95% of CWs are lower than this number of characters
        target_text = toot[1][:56]
        if not target_text:
            continue
        if target_text:
            num_with_cw += 1
        # We use "\x02" (ascii "start of text") as the "start sequence" character
        # for the targets, and "\x03" (ascii "end of text") as "end sequence" character.
        # \t and \n are used in the data so that's no-go
        target_text = '\x02' + target_text + '\x03'
        # Check for invalid characters
        for i, char in enumerate(input_text):
            if char not in characters:
                # Invalid character 
                # We choose \x05 as an 'invalid character' character somewhat
                # arbitrarily
                input_text = input_text[:i] + '\x05' + input_text[i+1:]
        # Again for CWs
        for i, char in enumerate(target_text):
            if char not in characters:
                target_text = target_text[:i] + '\x05' + target_text[i+1:]
        input_texts.append(input_text)
        target_texts.append(target_text)

    del raw_toots

    print("Percent of toots with CWs:", int(100 * num_with_cw / actual_num_samples))
    print()

    characters = sorted(list(characters))
    num_tokens = len(characters)
    input_lengths = [len(txt) for txt in input_texts]
    max_encoder_seq_length = max(input_lengths)
    lengths = [len(txt) for txt in target_texts]
    max_decoder_seq_length = max(lengths)

    print('Number of samples:', len(input_texts))
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    print('Average sequence length for inputs:', int(np.average(input_lengths)))
    print('StdDev sequence length for inputs:', int(np.std(input_lengths)))
    print('Average sequence length for outputs:', int(np.average(lengths)))
    print('StdDev sequence length for outputs:', int(np.std(lengths)))

    token_index = dict(
        [(char, i) for i, char in enumerate(characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, token_index[char]] = 1.

    return (encoder_input_data, decoder_input_data, decoder_target_data, token_index, input_texts)

def decode_sequence(encoder_model, decoder_model, token_index, input_seq):
    reverse_input_char_index = dict(
        (i, char) for char, i in token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in token_index.items())

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, len(token_index)))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, token_index['\x02']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        if sampled_char == '\x05':
            sampled_char = 'X' # Display an X it's easier to read
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\x03' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, len(token_index)))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

