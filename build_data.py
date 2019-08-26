#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build dataset for training and evluation. Remember to run this file before model training.

   Author: Meng Cao
"""

from model.config import Config
from model.data_utils import PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING, NUMBER, PRONOUNS, PRONOUNS_UPPER, get_processing_word, Dataset, get_vocabs, get_glove_vocab, load_vocab, write_vocab, export_trimmed_glove_vectors, get_char_vocab

def add_special_tokens(vocab):
    """Processing the vocab, add special tokens
    
    Args:
        vocab: list of words, string
    """
    assert not PAD_TOKEN in vocab
    assert not UNKNOWN_TOKEN in vocab
    assert not START_DECODING in vocab
    assert not STOP_DECODING in vocab
    assert not NUMBER in vocab

    vocab.insert(0, PAD_TOKEN) # PAD_TOKEN has index 0
    vocab.insert(1, UNKNOWN_TOKEN)
    vocab.insert(2, START_DECODING)
    vocab.insert(3, STOP_DECODING)
    vocab.insert(4, NUMBER)

def move_pronouns(vocab):
    """Move pronouns into the end of the list
    """
    pronouns_in_vocab = []
    for i, w in enumerate(vocab):
        if w in PRONOUNS:
            pronouns_in_vocab += [vocab.pop(i)]
    vocab.extend(pronouns_in_vocab)
    return pronouns_in_vocab

def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(load=False)
    pw_function = get_processing_word(lowercase=True)

    # Generators
    dev   = Dataset(config.filename_dev, processing_word=pw_function)
    test  = Dataset(config.filename_test, processing_word=pw_function)
    train = Dataset(config.filename_train, processing_word=pw_function)

    # Build Words
    vocab_words = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.filename_glove)

    vocab = vocab_words & vocab_glove
    vocab = list(vocab)
    pronouns_in_vocab = move_pronouns(vocab)
    write_vocab(pronouns_in_vocab, config.filename_pronouns)
    
    # add START, STOP, PAD, UNK and NUM tokens into the list
    add_special_tokens(vocab)
    assert PAD_TOKEN == vocab[0]
    assert UNKNOWN_TOKEN in vocab

    # Save vocab
    write_vocab(vocab, config.filename_words)

    # Trim GloVe Vectors
    vocab, _ = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
        config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = Dataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    vocab_chars = list(vocab_chars)
    vocab_chars.insert(0, PAD_TOKEN)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
