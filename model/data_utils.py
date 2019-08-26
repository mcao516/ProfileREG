# -*- coding: utf-8 -*-

"""This python file contains mothods for data preparation.
   
   Author: Meng Cao
"""

import numpy as np
import torch
import random
import pickle

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
NUMBER = "[NUM]" # special token for all numbers

PRONOUNS = ['I', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves', 'yourselves', 'they', 'them', 'their', 'theirs', 'themselves']

PRONOUNS_UPPER = ['You', 'She', 'He', 'It', 'We', 'They', 'My', 'Your', 'His', 'Her', 'Its', 'Our', 'Their', 'Mine', 'Yours', 'Hers', 'Ours', 'Theirs']


class Dataset(object):
    """For build dataset

    __iter__ method yields a tuple (pre_context, pos_context, description, refex)
        pre_context: list of raw words
        pos_context: list of raw words
        description: list of raw words
        refex: list of raw words

    Example:
        ```python
        data = Dataset(filename)
        for pre_context, pos_context, description, refex in data:
            pass
        ```

    """
    def __init__(self, dirname, processing_word=None, max_iter=None):
        """
        Args:
            dirname: path to the dir, in that dir must have 'entity.txt', 
                'refex.txt', 'pre_context.txt', 'pos_context.txt', 
                'description.txt'
            processing_word: function, processing word
            max_iter: (optional) max number of sentences to yield

        """
        self.dirname = dirname
        self.processing_word = processing_word
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        all_data = pickle.load(open(self.dirname, 'rb'))
        assert type(all_data) == type([])

        for ref in all_data:
            niter += 1
            if self.max_iter is not None and niter > self.max_iter:
                break

            _refex, _pre_c, _pos_c, _des = ref['refex'], ref['pre_context'], ref['pos_context'], ref['profile']
            
            # processing refex
            if self.processing_word is not None:
                refex = [self.processing_word(w) for w in _refex.split()]
                pre_c = [self.processing_word(w) for w in _pre_c.split()]
                pos_c = [self.processing_word(w) for w in _pos_c.split()]
                des   = [self.processing_word(w) for w in _des.split()]
                yield pre_c, pos_c, des, refex
            else:
                refex = [_ for _ in _refex.split()]
                pre_c = [_ for _ in _pre_c.split()]
                pos_c = [_ for _ in _pos_c.split()]
                des   = [_ for _ in _des.split()]
                yield pre_c, pos_c, des, refex
        
    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


class REGDataset(object):
    """Class that iterates over REG Dataset

    __iter__ method yields a tuple (pre_context, pos_context, description, refex)
        pre_context: list of raw words
        pos_context: list of raw words
        description: list of raw words
        refex: list of raw words
        
    If config is not None, return Example object.

    Example:
        ```python
        data = REGDataset(filename)
        for pre_context, pos_context, description, refex in data:
            pass
        ```

    """
    def __init__(self, dirname, config=None, max_iter=None):
        """
        Args:
            dirname: path to the dir, in that dir must have 'entity.txt', 
                'refex.txt', 'pre_context.txt', 'pos_context.txt', 
                'description.txt'
            config: (optional) Config object.
            max_iter: (optional) max number of sentences to yield

        """
        self.dirname = dirname
        self.config = config
        self.max_iter = max_iter
        self.length = None

        # read data
        self.data = []
        self._read_data()

    def __iter__(self):
        for example in self.data:
            yield example
        
    def __len__(self):
        if self.length is None:
            self.length = len(self.data)
        return self.length

    def shuffle(self):
        random.shuffle(self.data)

    def _read_data(self):
        niter = 0
        all_data = pickle.load(open(self.dirname, 'rb'))
        assert type(all_data) == type([])

        for ref in all_data:
            if self.max_iter is not None and niter > self.max_iter:
                    break
            # processing refex
            if ref['refex'][:4] == 'eos ' and ref['refex'][-4:] == ' eos':
                refex = ref['refex'][4: -4]
            else:
                refex = ref['refex']
            self.data.append(Example((ref['pre_context'], ref['pos_context']), 
                                      ref['profile'], refex, self.config))
            niter += 1

# class REGDataset(object):
#     """Class that iterates over REG Dataset

#     __iter__ method yields a tuple (pre_context, pos_context, description, refex)
#         pre_context: list of raw words
#         pos_context: list of raw words
#         description: list of raw words
#         refex: list of raw words
        
#     If config is not None, return Example object.

#     Example:
#         ```python
#         data = REGDataset(filename)
#         for pre_context, pos_context, description, refex in data:
#             pass
#         ```

#     """
#     def __init__(self, dirname, config=None, max_iter=None):
#         """
#         Args:
#             dirname: path to the dir, in that dir must have 'entity.txt', 
#                 'refex.txt', 'pre_context.txt', 'pos_context.txt', 
#                 'description.txt'
#             config: (optional) Config object.
#             max_iter: (optional) max number of sentences to yield

#         """
#         self.dirname = dirname
#         self.config = config
#         self.max_iter = max_iter
#         self.length = None

#     def __iter__(self):
#         niter = 0
#         wiki_id_file = open(self.dirname+'/entity.txt', 'r', encoding='utf-8')
#         refex_file   = open(self.dirname+'/refex.txt', 'r', encoding='utf-8')
#         pre_context_file = open(self.dirname+'/pre_context.txt', 'r', encoding='utf-8')
#         pos_context_file = open(self.dirname+'/pos_context.txt', 'r', encoding='utf-8')
#         description_file = open(self.dirname+'/description.txt', 'r', encoding='utf-8')
        
#         for _wiki_id, _refex, _pre_c, _pos_c, _des in zip(wiki_id_file, 
#             refex_file, pre_context_file, pos_context_file, description_file):
#             _wiki_id, _refex, _pre_c, _pos_c, _des = _wiki_id.strip(), \
#             _refex.strip(), _pre_c.strip(), _pos_c.strip(), _des.strip()
            
#             if _wiki_id != '':
#                 niter += 1
#                 if self.max_iter is not None and niter > self.max_iter:
#                     break

#                 if self.config.use_context_words:
#                     pre_last_word = _pre_c.split()[-1]
#                     pos_first_word = _pos_c.split()[0]
#                     _refex = pre_last_word + ' ' + _refex + ' ' + pos_first_word
                
#                 # processing refex
#                 yield Example((_pre_c, _pos_c), _des, _refex, self.config)
        
#         refex_file.close()
#         wiki_id_file.close()
#         pre_context_file.close()
#         pos_context_file.close()
#         description_file.close()
        
#     def __len__(self):
#         """Iterates once over the corpus to set and store length"""
#         if self.length is None:
#             self.length = 0
#             for _ in self:
#                 self.length += 1
        
#         return self.length


def article2ids(article_words, word2id, vocab_size):
    """Map the article words to their ids. Also return a list of OOVs in the article.

       Args:
           article_words: list of words (strings)
           config: config object

       Returns:
           ids: A list of word ids (integers); OOVs are represented by their 
           temporary article OOV number. If the vocabulary size is 50k and the 
           article has 3 OOVs, then these temporary OOV numbers will be 50000, 
           50001, 50002.
          
           oovs: A list of the OOV words in the article (strings), in the order 
           corresponding to their temporary article OOV numbers.
    """
    ids, oovs = [], []
    for w in article_words:
        # i = config.processing_word(w)
        if not w in word2id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab_size + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(word2id[w])
    return ids, oovs


def refex2ids(refex_words, word2id, article_oovs, vocab_size):
    """Map the refex words to their ids. In-article OOVs are mapped to their       temporary OOV numbers.
    
       Args:
           refex_words: list of words (strings)
           vocab: Vocabulary object
           article_oovs: list of in-article OOV words (strings), in the order 
                         corresponding to their temporary article OOV numbers
       Returns:
           ids: List of ids (integers). In-article OOV words are mapped to 
                their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id.
    """
    ids = []
    unk_id = word2id[UNKNOWN_TOKEN]
    for w in refex_words:
        if not w in word2id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                # Map to its temporary article OOV number
                vocab_idx = vocab_size + article_oovs.index(w) 
                ids.append(vocab_idx)
            else: # If w is an out-of-article OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(word2id[w])
    return ids


class Example(object):
    """Class representing a single train/val/test example for referring 
        expression generation.
    """

    def __init__(self, context, description, refex, config):
        """Initializes the Example, performing str-to-id and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        Args:
            context: tuple of strings; Reresent pre- and pos-context.
                    Each token is separated by a single space.
            description: string; description of the target entity.
            refex: string; referring expression.
            config: config object, used to determine the max sequence lengths and convert word str to id
        """
        self.config = config
        self.pad_id = self.config.word2id[PAD_TOKEN]

        # Process the context
        assert len(context) == 2
        pre_c, pos_c = context[0].split(), context[1].split()
        if self.config.reverse_pos_context:
            pos_c.reverse()
        if len(pre_c) > config.max_enc_context:
            pre_c = pre_c[:config.max_enc_context]
        if len(pos_c) > config.max_enc_context:
            pos_c = pos_c[:config.max_enc_context]
        # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input_prec = [config.processing_word(w) for w in pre_c]
        self.enc_input_prec_len = len(pre_c)
        self.enc_input_posc = [config.processing_word(w) for w in pos_c]
        self.enc_input_posc_len = len(pos_c)

        # Process the description
        des_words = description.split()
        if len(des_words) > config.max_enc_description:
            des_words = des_words[:config.max_enc_description]
        # store the length after truncation but before padding
        self.enc_input_des_len = len(des_words)
        # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input_des = [config.processing_word(w) for w in des_words]
        # Store a version of the enc_input where in-article OOVs are 
        # represented by their temporary OOV id;
        self.enc_input_des_extended, self.des_oovs = article2ids(des_words, 
            config.word2id, config.nwords)

        # Process the referrring expression
        start_decoding = config.word2id[START_DECODING]
        stop_decoding  = config.word2id[STOP_DECODING]
        refex_words = refex.split() # list of strings
        # list of word ids; OOVs are represented by the id for UNK token
        refex_ids = [config.word2id[w] if w in config.word2id else config.word2id[UNKNOWN_TOKEN] for w in refex_words]
        # Get the decoder input sequence and target sequence
        self.dec_input, _ = self.get_dec_inp_targ_seqs(refex_ids, 
            config.max_dec_steps, start_decoding, stop_decoding)
        # Get a verison of the refex where in-article OOVs are represented
        # by their temporary article OOV id
        refex_ids_extended = refex2ids(refex_words, config.word2id, 
            self.des_oovs, config.nwords)
        # Overwrite decoder target sequence so it uses the temp article OOV ids
        _, self.target = self.get_dec_inp_targ_seqs(refex_ids_extended, 
            config.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # Store the original strings
        self.original_context = context
        self.original_description = description
        self.original_refex = refex

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """Given the refex as a sequence of tokens, return the 
           input sequence for the decoder, and the target sequence which 
           we will use to calculate loss. 

           The sequence will be truncated if it is longer than max_len. 
           The input sequence must start with the start_id and  the target 
           sequence must end with the stop_id (but not if it's been truncated).

           Args:
               sequence: List of ids (integers)
               max_len: integer
               start_id: integer
               stop_id: integer

           Returns:
               inp: sequence length <=max_len starting with start_id
               target: sequence same length as input, ending with stop_id only if there was no truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len: # truncate
            inp = inp[:max_len]
            target = target[:max_len] # no end_token
        else: # no truncation  
            target.append(stop_id) # end token
        assert len(inp) == len(target)
        return inp, target    

    # def pad_context(self, pre_max_len, pos_max_len):
    #     """Pad the encoder input sequence with pad_id up to max_len.
    #     """
    #     pad_id = self.config.processing_word(PAD_TOKEN)
    #     while len(self.enc_input_prec) < pre_max_len:
    #         self.enc_input_prec.append(pad_id)
    #     while len(self.enc_input_posc) < pos_max_len:
    #         self.enc_input_posc.append(pad_id)

    # def pad_description(self, max_len):
    #     """Pad the encoder input sequence with pad_id up to max_len.
    #     """
    #     pad_id = self.config.processing_word(PAD_TOKEN)
    #     while len(self.enc_input_des) < max_len:
    #         self.enc_input_des.append(pad_id)
    #     while len(self.enc_input_des_extended) < max_len:
    #         self.enc_input_des_extended.append(pad_id)

    def pad_decoder_inp_targ(self):
        """Pad decoder input and target sequences with pad_id up to max_len.
        """
        pad_id = self.config.word2id[PAD_TOKEN]
        while len(self.dec_input) < self.config.max_dec_steps:
            self.dec_input.append(pad_id)
        while len(self.target) < self.config.max_dec_steps:
            self.target.append(pad_id)


class Batch(object):
    """Class representing a minibatch of train/val/test examples for text     
       summarization.
    """

    def __init__(self, example_list):
        """Turns the example_list into a Batch object.

        Args:
            example_list: List of Example objects
        """
        self.batch_size = len(example_list)
        assert self.batch_size > 0
        # initialize the input context to the encoder
        self.init_encoder_context(example_list)
        # initialize the input description to the encoder 
        self.init_encoder_des(example_list)
        # initialize the input and targets refex for the decoder
        self.init_decoder_seq(example_list)
        # store the original strings
        self.store_orig_strings(example_list) 

    def init_encoder_context(self, example_list):
        """Initializes the following:
           
        Args:
           self.prec_word_ids: [batch_size, max_seq_len]

           self.prec_seq_lens: [batch_size]
        
           self.prec_char_ids: [batch_size, max_seq_len, max_word_len]
           
           self.prec_word_lens: [batch_size, max_seq_len]

           self.prec_seq_mask: [batch_size, max_seq_len]

           self.posc_word_ids: [batch_size, max_seq_len]

           self.posc_seq_lens: [batch_size]
        
           self.posc_char_ids: [batch_size, max_seq_len, max_word_len]
           
           self.posc_word_lens: [batch_size, max_seq_len]

           self.posc_seq_mask: [batch_size, max_seq_len]
        """
        prec_batch, posc_batch = [], []
        for ex in example_list:
            pad_id = ex.pad_id
            prec, posc = ex.enc_input_prec, ex.enc_input_posc
            if type(prec[0]) == tuple:
                prec, posc = zip(*prec), zip(*posc)
            prec_batch += [prec]
            posc_batch += [posc]
        
        # pad pre-context
        char_ids, word_ids = zip(*prec_batch)
        # prec_word_ids: [batch_size, max_seq_len]
        # prec_seq_lens: [batch_size]
        self.prec_word_ids, self.prec_seq_lens = pad_sequences(word_ids, pad_id)
        assert len(self.prec_word_ids) == self.batch_size
        assert len(self.prec_seq_lens) == self.batch_size
        # prec_char_ids: [batch_size, max_seq_len, max_word_len]
        # prec_word_lens: [batch_size, max_seq_len]
        self.prec_char_ids, self.prec_word_lens = pad_sequences(char_ids, 
            pad_tok=pad_id, nlevels=2)
        assert len(self.prec_char_ids) == self.batch_size
        assert len(self.prec_word_lens) == self.batch_size
        # build pre-context mask
        max_prec_len = len(self.prec_word_ids[0])
        self.prec_seq_mask = []
        for i in self.prec_seq_lens:
            self.prec_seq_mask += [[1]*i+[0]*(max_prec_len-i)]

        # pad pos-context
        char_ids, word_ids = zip(*posc_batch)
        # posc_word_ids: [batch_size, max_seq_len]
        # posc_seq_lens: [batch_size]
        self.posc_word_ids, self.posc_seq_lens = pad_sequences(word_ids, pad_id)
        assert len(self.posc_word_ids) == self.batch_size
        assert len(self.posc_seq_lens) == self.batch_size
        # posc_char_ids: [batch_size, max_seq_len, max_word_len]
        # posc_word_lens: [batch_size, max_seq_len]
        self.posc_char_ids, self.posc_word_lens = pad_sequences(char_ids, 
            pad_tok=pad_id, nlevels=2)
        assert len(self.posc_char_ids) == self.batch_size
        assert len(self.posc_word_lens) == self.batch_size
        # build pos-context mask
        max_posc_len = len(self.posc_word_ids[0])
        self.posc_seq_mask = []
        for i in self.posc_seq_lens:
            self.posc_seq_mask += [[1]*i+[0]*(max_posc_len-i)]

        # # numpy arrays for pre-context ids
        # self.prec_seq_ids = np.asarray(prec_word_ids, dtype=np.int64)
        # self.prec_seq_lens = np.asarray(prec_seq_lens, dtype=np.int64)
        # self.prec_char_ids = np.asarray(prec_char_ids, dtype=np.int64)
        # self.prec_word_lens = np.asarray(prec_word_lens, dtype=np.int64)
        # self.prec_seq_mask = np.asarray(prec_seq_mask, dtype=np.int64)

        # # numpy arrays for pos-context ids
        # self.posc_seq_ids = np.asarray(posc_word_ids, dtype=np.int64)
        # self.posc_seq_lens = np.asarray(posc_seq_lens, dtype=np.int64)
        # self.posc_char_ids = np.asarray(posc_char_ids, dtype=np.int64)
        # self.posc_word_lens = np.asarray(posc_word_lens, dtype=np.int64)
        # self.posc_seq_mask = np.asarray(posc_seq_mask, dtype=np.int64)

    def init_encoder_des(self, example_list):
        """Initializes the following:

        Args:
            self.des_word_ids: [batch_size, max_seq_len], list of list of ids of descripton words, the out-of-vocabulary words are represented by UNK token.

            self.des_seq_lens: [batch_size], list of integers of length of each description.

            self.des_char_ids: [batch_size, max_seq_len, max_word_len], integers represent ids of characters in description.
            
            self.des_word_lens: [batch_size, max_seq_len]
        """
        des_batch, des_extented_batch = [], []
        for ex in example_list:
            pad_id = ex.pad_id
            des, des_extented = ex.enc_input_des, ex.enc_input_des_extended
            if type(des[0]) == tuple:
                des = zip(*des)
            des_batch += [des]
            des_extented_batch += [des_extented]

        # pad description
        char_ids, word_ids = zip(*des_batch)
        self.des_word_ids, self.des_seq_lens = pad_sequences(word_ids, pad_id)
        assert len(self.des_word_ids) == self.batch_size
        assert len(self.des_seq_lens) == self.batch_size
        self.des_char_ids, self.des_word_lens = pad_sequences(char_ids, 
            pad_tok=pad_id, nlevels=2)
        assert len(self.des_char_ids) == self.batch_size
        assert len(self.des_word_lens) == self.batch_size
        # build description mask
        max_des_len = len(self.des_word_ids[0])
        self.des_mask = []
        for i in self.des_seq_lens:
            self.des_mask += [[1]*i + [0]*(max_des_len - i)]

        # pad extented description
        # des_extented_batch: list of list of ids [[2, 13, ...], ...]
        self.des_extented_ids, _ = pad_sequences(des_extented_batch, pad_id)
        assert len(self.des_extented_ids) == self.batch_size

        # Determine the max number of in-article OOVs in this batch
        self.max_des_oovs = max([len(ex.des_oovs) for ex in example_list])
        # Store the in-article OOVs themselves
        self.des_oovs = [ex.des_oovs for ex in example_list]

    def init_decoder_seq(self, example_list):
        """Initializes the following:

        self.dec_batch: numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.

        self.target_batch: numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.

        self.dec_padding_mask: numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.

        """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ()

        # Initialize the numpy arrays.
        max_dec_steps = len(example_list[0].dec_input)
        self.dec_batch = np.zeros((self.batch_size, max_dec_steps), 
            dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, max_dec_steps), 
            dtype=np.int32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, max_dec_steps), 
            dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""
        # Store the original strings
        self.original_description = [ex.original_description for ex in example_list]
        self.original_context = [ex.original_context for ex in example_list]
        self.original_refex = [ex.original_refex for ex in example_list]


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 1,
                max_length_sentence)
        # sequence_length, _ = _pad_sequences(sequence_length, 0,
        #         max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of Example object
        minibatch_size: (int)

    Yields:
        Batch object

    """
    ex_list = []
    for ex in data:
        if len(ex_list) == minibatch_size:
            yield Batch(ex_list)
            ex_list = []

        ex_list.append(ex)

    if len(ex_list) != 0:
        yield Batch(ex_list)


def get_processing_word(vocab_words=None, vocab_chars=None,
                        lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUMBER

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNKNOWN_TOKEN]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    for dataset in datasets:
        for pre_c, pos_c, des, refex in dataset:
            vocab_words.update(pre_c, pos_c, des, refex)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    word2id = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            word2id[word] = idx
    id2word = {v: k for k, v in word2id.items()}
    assert len(word2id) == len(id2word)
    return word2id, id2word


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    with np.load(filename) as data:
        return data["embeddings"]
        

def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for pre_c, pos_c, des, refex in dataset:
        for word in pos_c:
            vocab_char.update(word)
        for word in refex:
            vocab_char.update(word)
        for word in des:
            vocab_char.update(word)

    return vocab_char
