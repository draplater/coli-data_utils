import numpy as np


def lookup_list(tokens_itr, token_dict, padded_length, default):
    ret = np.zeros((padded_length,), dtype=np.int32)
    for idx, word in enumerate(tokens_itr):
        ret[idx] = token_dict.get(word, default)

    return ret


def lookup_characters(words_itr, char_dict, padded_length, default, max_word_length=20):
    chars_int = np.zeros((padded_length, max_word_length), dtype=np.int32)

    for idx, word in enumerate(words_itr):
        if len(word) > max_word_length:
            word = word[:9] + ".." + word[(len(word) - 9):]
        for c_idx, character in enumerate(word):
            chars_int[idx, c_idx] = char_dict.get(character, default)

    return chars_int
