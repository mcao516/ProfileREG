#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ProfileREG model for referring expression generation (REG).

   Author: Meng Cao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import os

from .config import Config
from .general_utils import Progbar
from .data_utils import minibatches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding_layer(nn.Module):
    """Embedding layer of the model.
    """
    def __init__(self, vocab_size, word_dim, char_size, char_dim, char_hidden_dim, pre_trained=None, padding_idx=0, drop_out=0.5):
        """Initialize embedding layer. If pre_trained is provided, 
           initialize the model using pre-trained embeddings.
        """
        super(Embedding_layer, self).__init__()
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.word_dim = word_dim
        self.char_dim = char_dim
        self.char_hidden_dim = char_hidden_dim
        self.pre_trained = pre_trained
        self.padding_idx = padding_idx
        self.drop_out = drop_out
        
        self._build_model(self.drop_out) # build model...
    
    def _build_model(self, drop_out):
        self.word_embedding = nn.Embedding(self.vocab_size, self.word_dim,
                                           padding_idx=self.padding_idx)
        self.char_embedding = nn.Embedding(self.char_size, self.char_dim,
                                           padding_idx=self.padding_idx)
        
        self.char_rnn = nn.LSTM(self.char_dim, self.char_hidden_dim,
                                batch_first=True, bidirectional=True)

        self.char_proj = nn.Linear(self.char_hidden_dim*2, self.char_hidden_dim)
        
        # apply pre-trained embeddings
        if self.pre_trained is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.pre_trained))
        # add dropout layer
        self.dropout = nn.Dropout(p=drop_out)
        
    def forward(self, word_ids, char_ids=None, word_lens=None):
        """
        Args:
            word_ids: [batch_size, max_seq_len]
            char_ids: [batch_size, max_seq_len, max_word_len]
            word_lens: [batch_size, max_seq_len]
        Return:
            embedded: [batch_size, max_seq_len, embedding_dim]
        """
        # word_embeddings: [batch_size, max_seq_len, word_dim]
        word_embeddings = self.word_embedding(word_ids)
        # word_embeddings = self.dropout(word_embeddings)
        # no character-level embeddings
        if char_ids is None:
            return word_embeddings
        
        batch_size, max_seq_len, max_word_len = char_ids.size()
        # char_embedded: [batch_size, max_seq_len, max_word_len, char_dim]
        char_embedded = self.char_embedding(char_ids)
        # char_embedded = self.dropout(char_embedded)
        char_embedded = char_embedded.view(batch_size*max_seq_len, max_word_len, -1)
        
        assert word_lens.size(0) == batch_size and word_lens.size(1) == max_seq_len
        word_lens = word_lens.view(batch_size*max_seq_len)
        lengths_sorted, idx_sort = torch.Tensor.sort(word_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        assert char_embedded.size(0) == idx_sort.size(0)
        char_embedded = char_embedded.index_select(0, idx_sort)
        char_packed = pack_padded_sequence(char_embedded, lengths_sorted, batch_first=True)
        
        # h_n: [2, batch_size*max_seq_len, char_hidden_dim]
        _, (h_n, _) = self.char_rnn(char_packed)
        assert h_n[0].size(0) == idx_unsort.size(0)
        fw_hn = h_n[0].index_select(0, idx_unsort)
        bw_hn = h_n[1].index_select(0, idx_unsort)
        # idx_unsort_expand = idx_unsort.view(-1, 1).expand(batch_size*max_seq_len, h_n[0].size(-1))
        # fw_hn = h_n[0].gather(0, idx_unsort_expand)
        # bw_hn = h_n[1].gather(0, idx_unsort_expand)
        assert fw_hn.size(0) == batch_size*max_seq_len    
    
        # char_hidden: [batch_size, max_seq_len, 2*char_hidden_dim]
        char_hiddens = torch.cat((fw_hn, bw_hn), -1).view(batch_size, max_seq_len, -1)
        assert char_hiddens.size(2) == self.char_hidden_dim*2

        char_hiddens = torch.tanh(self.char_proj(char_hiddens))
        # char_hiddens = self.dropout(char_hiddens)
        assert char_hiddens.size(2) == self.char_hidden_dim

        # [batch_size, max_seq_len, word_dim+char_hidden_dim]
        final_embeddings = torch.cat((word_embeddings, char_hiddens), -1)
        final_embeddings = self.dropout(final_embeddings)
        return final_embeddings


# encoder net for the article
class ContextEncoder(nn.Module):
    """Sinal directional LSTM network, encoding pre- and pos-context.
    """
    def __init__(self, input_size, hidden_size, wordEmbed, drop_out=0.5):
        """Initialize Context Encoder
        Args:
            input_size: input embedding size
            hidden_size: output hidden size
        """
        super(ContextEncoder,self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.word_embed  = wordEmbed # embedding layer
        self.drop_out = drop_out
        
        self._build_model(self.drop_out)
    
    def _build_model(self, drop_out):
        """Build context-encoder model"""
        self.pre_rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.pos_rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.output_cproj = nn.Linear(self.hidden_size*4, self.hidden_size)
        self.output_hproj = nn.Linear(self.hidden_size*4, self.hidden_size)
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, _pre, _pos):
        """Encoding context sequences
        
        Args:
            _pre: (prec_word_ids, prec_seq_lens, prec_char_ids, prec_word_lens)
            _pos: (posc_word_ids, posc_seq_lens, posc_char_ids, posc_word_lens)
        """
        prec_word_ids, prec_seq_lens, prec_char_ids, prec_word_lens = _pre
        posc_word_ids, posc_seq_lens, posc_char_ids, posc_word_lens = _pos
        # embed_pre: [batch, max_seq_len, word_dim+char_hidden*2]
        embed_pre = self.word_embed(prec_word_ids, prec_char_ids, prec_word_lens)
        # embed_pos: [batch, max_seq_len, word_dim+char_hidden*2]
        embed_pos = self.word_embed(posc_word_ids, posc_char_ids, posc_word_lens)
        
        # sort lengths
        prec_lens_sorted, pre_idx_sort = torch.sort(prec_seq_lens, dim=0, descending=True)
        posc_lens_sorted, pos_idx_sort = torch.sort(posc_seq_lens, dim=0, descending=True)
        _, pre_idx_unsort = torch.sort(pre_idx_sort, dim=0)
        _, pos_idx_unsort = torch.sort(pos_idx_sort, dim=0)
        # sort embedded sentences
        embed_pre = embed_pre.index_select(0, pre_idx_sort)
        embed_pos = embed_pos.index_select(0, pos_idx_sort)
        
        pre_packed = pack_padded_sequence(embed_pre, prec_lens_sorted, batch_first=True)
        pos_packed = pack_padded_sequence(embed_pos, posc_lens_sorted, batch_first=True)
        
        # pre_state: ([2, batch_size, hidden], [2, batch_size, hidden])
        _, pre_state = self.pre_rnn(pre_packed)
        _, pos_state = self.pos_rnn(pos_packed)
        
        # restore to the initial order
        pre_h, pre_c = torch.cat((pre_state[0][0], pre_state[0][1]), -1), torch.cat((pre_state[1][0], pre_state[1][1]), -1)
        pos_h, pos_c = torch.cat((pos_state[0][0], pos_state[0][1]), -1), torch.cat((pos_state[1][0], pos_state[1][1]), -1)
        # pre_idx_resort = pre_idx_unsort.view(-1, 1).expand(pre_h.size(0), pre_h.size(1))
        # pos_idx_resort = pos_idx_unsort.view(-1, 1).expand(pos_h.size(0), pos_h.size(1))
        # pre_h = pre_h.gather(0, pre_idx_resort)
        # pre_c = pre_c.gather(0, pre_idx_resort)
        # pos_h = pos_h.gather(0, pos_idx_resort)
        # pos_c = pos_c.gather(0, pos_idx_resort)
        pre_h = pre_h.index_select(0, pre_idx_unsort)
        pre_c = pre_c.index_select(0, pre_idx_unsort)
        pos_h = pos_h.index_select(0, pos_idx_unsort)
        pos_c = pos_c.index_select(0, pos_idx_unsort)
        
        # final_hidd_proj/final_cell_proj: [batch_size, hidden]
        final_hidd_proj = self.tanh(self.output_hproj(torch.cat((pre_h, pos_h), 1)))
        final_cell_proj = self.tanh(self.output_cproj(torch.cat((pre_c, pos_c), 1)))
        
        del embed_pre, embed_pos, pre_packed, pos_packed
        del pre_state, pos_state
        del pre_h, pre_c, pos_h, pos_c

        final_hidd_proj = self.dropout(final_hidd_proj)
        final_cell_proj = self.dropout(final_cell_proj)

        return final_hidd_proj, final_cell_proj


# encoder net for the article
class BidirectionalEncoder(nn.Module):
    """Bidirectional LSTM encoder.
    """
    def __init__(self, input_size, hidden_size, wordEmbed, drop_out=0.5):
        super(BidirectionalEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.word_embed = wordEmbed # embedding layer

        self._build_model(drop_out) # build model...
        
    def _build_model(self, drop_out):
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, 
            batch_first=True, bidirectional=True)
        self.output_cproj = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.output_hproj = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_word_ids, input_seq_lens, input_seq_mask, 
                input_char_ids, input_word_lens):
        """Encode description, return outputs and the last hidden state
           
        Args:
            input_word_ids: [batch_size, max_seq_len]
            input_seq_lens: [batch_size]
            input_seq_mask: [batch_size, max_seq_len]
            input_char_ids: [batch_size, max_seq_len, max_word_len]
            input_word_lens: [batch_size, max_seq_len]
        """
        # _input: [batch, max_seq_len]
        batch_size, max_len = input_word_ids.size(0), input_word_ids.size(1)
        # embed_wd: [batch, max_seq_len, word_dim + char_dim*2]
        embed_wd = self.word_embed(input_word_ids, input_char_ids, input_word_lens)
        
        # sorting the batch for packing
        lengths_sorted, idx_sort = torch.sort(input_seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        embed_wd = embed_wd.index_select(0, idx_sort)
        
        input_packed = pack_padded_sequence(embed_wd, lengths_sorted, batch_first=True)
        
        # outputs: [batch, max_seq_len, hidden_size * 2]
        # final_state: ([2, batch, hidden_size], [2, batch, hidden_size])
        outputs_packed, final_state = self.encoder(input_packed)
        outputs_padded, _ = pad_packed_sequence(outputs_packed, batch_first=True)
        outputs = outputs_padded.index_select(0, idx_unsort)
        
        h_n = final_state[0].index_select(1, idx_unsort)
        c_n = final_state[1].index_select(1, idx_unsort)
        assert outputs.size(0) == batch_size
        assert outputs.size(1) == max_len
        assert outputs.size(2) == self.hidden_size * 2
        
        # inv_mask: [batch, max_seq_len, hidden_size * 2]
        mask = input_seq_mask.eq(0).detach()
        inv_mask = mask.eq(0).unsqueeze(2).expand(batch_size, max_len, 
            self.hidden_size * 2).float().detach()
        hidden_out = outputs * inv_mask
        
        # final_hidd_proj: [batch_size, hidden_size]
        final_hidd_proj = self.output_hproj(torch.cat((h_n[0], h_n[1]), 1))
        final_cell_proj = self.output_cproj(torch.cat((c_n[0], c_n[1]), 1))

        del embed_wd, input_packed

        # apply dropout
        hidden_out = self.dropout(hidden_out)
        final_hidd_proj = self.dropout(final_hidd_proj)
        final_cell_proj = self.dropout(final_cell_proj)

        return hidden_out, final_hidd_proj, final_cell_proj, mask


class Hypothesis(object):
    def __init__(self, token_id, hidden_state, cell_state, log_prob):
        self.full_prediction = token_id # list
        self._h = hidden_state
        self._c = cell_state
        self.log_prob = log_prob
        self.survivability = self.log_prob/float(len(self.full_prediction))

    def extend(self, token_id, hidden_state, cell_state, log_prob):
        """Extend a beam path, add new token_id, update hidden and cell state,
           and modify the probility.
        """
        return Hypothesis(token_id=self.full_prediction + [token_id],
                          hidden_state=hidden_state, cell_state=cell_state,
                          log_prob= self.log_prob + log_prob)


# TODO Enhancement: Project input embedding with previous context vector for 
# current input
class PointerAttentionDecoder(nn.Module):
    """Pointer-generator attention decoder.
    """
    def __init__(self, input_size, hidden_size, vocab_size, wordEmbed):
        super(PointerAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.word_embed = wordEmbed
        
        self.beam_search = True # if doing beam search
        self.max_article_oov = None # max number of OOVs in a batch of data
        
        self._build_model() # build the model
        
    def _build_model(self):
        # lstm decoder
        self.decoderRNN = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        
        # params for attention
        # v tanh(W_h h + W_s s + b)
        self.Wh = nn.Linear(self.hidden_size*2, self. hidden_size*2)
        self.Ws = nn.Linear(self.hidden_size, self.hidden_size*2)
        self.v  = nn.Linear(self.hidden_size*2, 1)

        # parameters for p_gen
        # sigmoid(w_h h* + w_s s + w_x x + b)
        self.w_h = nn.Linear(self.hidden_size*2, 3) # attention context vector
        self.w_s = nn.Linear(self.hidden_size, 3) # hidden state
        self.w_x = nn.Linear(self.input_size, 3) # input vector
        self.w_c = nn.Linear(self.hidden_size, 3) # context encoder final hidden state
        
        # params for output proj
        self.V = nn.Linear(self.hidden_size*3, self.vocab_size)

        # dropout layer
        # self.dropout = nn.Dropout(p=0.5)

    def setValues(self, start_id, stop_id, unk_id, nprons, beam_size, min_decode=3, max_decode=10):
        # start/stop tokens
        self.start_id = start_id
        self.stop_id = stop_id
        self.unk_id = unk_id
        # decoding parameters
        self.nprons = nprons
        self.beam_size = beam_size
        self.min_length = min_decode
        self.max_decode_steps = max_decode
        
    def forward(self, enc_states, enc_final_state, enc_mask, article_inds, 
                _input, targets, dec_lens, dec_mask, decode=False):
        """enc_states [batch, max_seq_len, 2*hidden_size]:
               Output states of descirption bidirectional encoder.
           enc_final_states ([batch, hidden_size], ...):
               Final state of context encoder.
           enc_mask [batch_size, max_enc_len]:
               0 or 1 mask for descirption decoder output states.
           article_inds [batch_size, enc_seq_len]:
               Description encoder input with temporary OOV ids repalce 
               each UNK token
           _input [batch_size, dec_seq_len]:
               Decoder inputs, unk token for unknow words
           targets [batch_size, dec_seq_len]:
               Decoder targets, temporary OOV ids for unknow words
           dec_lens [batch_size]:
               Lengths of encoder inputs
           dec_mask [batch_size, des_seq_len]:
               Padding mask for encoder input
           decode Boolean:
               flag for train/eval mode
        """
        if decode is True:
            if self.beam_search:
                return self.decode(enc_states, enc_final_state, enc_mask, article_inds)
            else:
                return self.greedy_decoding(enc_states, enc_final_state, enc_mask, article_inds)
        
        # for attention calculation
        # enc_proj: [batch_size, max_enc_len, 2*hidden]
        batch_size, max_enc_len, enc_size = enc_states.size()
        enc_proj = self.Wh(enc_states.view(batch_size*max_enc_len, enc_size)).view(batch_size, max_enc_len, -1)	
        
        # embed_input: [batch_size, dec_seq_len, embedding_dim]
        embed_input = self.word_embed(_input)
        # state: ([1, batch_size, hidden_size], ...)
        state = enc_final_state[0].unsqueeze(0), enc_final_state[1].unsqueeze(0)
        # hidden: [batch_size, dec_seq_len, hidden_size] 
        hidden, _ = self.decoderRNN(embed_input, state)
        
        lm_loss = []
        
        max_dec_len = _input.size(1)
        # step through decoder hidden states
        for _step in range(max_dec_len):
            _h = hidden[:, _step, :] # _h: [batch_size, hidden_size]
            target = targets[:, _step].unsqueeze(1) # target: [batch_size, 1]
            # mask: [batch_size, 1]
            target_mask_0 = dec_mask[:, _step].unsqueeze(1) 
            # dec_proj: [batch_size, max_enc_len, 2*hidden_size]
            dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
            # dropout
            # enc_proj = self.dropout(enc_proj)
            # dec_proj = self.dropout(dec_proj)
            # attn_scores: [batch_size, max_enc_len]
            e_t = self.v(torch.tanh(enc_proj + dec_proj).view(batch_size*max_enc_len, -1))
            attn_scores = e_t.view(batch_size, max_enc_len)
            # mask to -INF before applying softmax
            attn_scores.masked_fill_(enc_mask, -float('inf'))
            # attn_scores: [batch_size, max_enc_len]
            attn_scores = F.softmax(attn_scores, 1)
            del e_t
            
            # context: [batch_size, 2*hidden_size]
            context = attn_scores.unsqueeze(1).bmm(enc_states).squeeze(1)
            # p_vocab: [batch_size, vocab_size]
            p_vocab = F.softmax(self.V(torch.cat((_h, context), 1)), 1)
            # dropout
            enc_final_state_proj = self.w_c(enc_final_state[0])
            _h_proj = self.w_s(_h)
            # p_switch: [batch_size, 3]
            p_switch = torch.nn.functional.softmax(self.w_h(context) + enc_final_state_proj + self.w_x(embed_input[:, _step, :]) + _h_proj, dim=1)
            p_switch = p_switch.view(-1, 3)

            p_gen = torch.cat((p_switch[:, 0].view(-1, 1).expand(batch_size, self.vocab_size-self.nprons), p_switch[:, 1].view(-1, 1).expand(batch_size, self.nprons)), dim=1)
            assert p_gen.size(0) == batch_size and p_gen.size(1) == self.vocab_size
            p_copy = p_switch[:, 2].view(-1, 1) # [batch_size, 1]

            # weighted_Pvocab: [batch_size, vocab_sze]
            weighted_Pvocab = p_gen * p_vocab 
            weighted_attn = p_copy * attn_scores # [batch_size, max_enc_len]
            assert weighted_attn.size(0) == batch_size and weighted_attn.size(1) == max_enc_len

            if self.max_article_oov > 0:
                # create OOV (but in-article) zero vectors
                ext_vocab = torch.zeros((batch_size, self.max_article_oov), 
                                        requires_grad=True, device=device)
                combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
                del ext_vocab
            else:
                combined_vocab = weighted_Pvocab

            del weighted_Pvocab, p_vocab
            assert article_inds.data.min() >= 0 and article_inds.data.max() < \
                (self.vocab_size + self.max_article_oov), \
                'Recheck OOV indexes! {}/{}'.format(self.max_article_oov, article_inds)

            # scatter article word probs to combined vocab prob.
            # No need to subtract, masked_fill_ 0 ?
            # article_inds_masked = article_inds.masked_fill_(enc_mask, 0)
            article_inds_masked = article_inds
            # combined_vocab: [batch_size, vocab_size + max_oov_num]
            combined_vocab = combined_vocab.scatter_add(1, article_inds_masked, weighted_attn)
            # output: [batch_size, 1]
            output = combined_vocab.gather(1, target) # target: [batch_size, 1]

            # unk_mask: [batch_size, 1]
            unk_mask = target.eq(self.unk_id).detach()
            output.masked_fill_(unk_mask, 1.0)

            lm_loss.append(output.log().mul(-1)*target_mask_0.float())
            
        # add individual losses
        total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens.float())
        return total_masked_loss

    def decode_step(self, enc_states, enc_h_n, state, _input, enc_mask, article_inds):
        """One step of decoding
        Args:
            enc_states: [batch, max_seq_len, hidden_size]
            enc_h_n: [1, hidden_size], last hidden state of context encoder hidden state
            state: [beam_size, hidden_size], previous time step hidden state
            _input: current time step input
        Returns:
            combined_vocab: [beam_size, vocab+extra_oov]
            _h, _c: ([beam_size, hidden_size], ...)
        """
        batch_size, max_enc_len, enc_size = enc_states.size()
        enc_proj = self.Wh(enc_states.view(batch_size*max_enc_len, enc_size)).view(batch_size, max_enc_len, -1)
        
        assert _input.max().item() < self.vocab_size, 'Word id {} is out of index'.format(_input.max().item())
        embed_input = self.word_embed(_input)
        _h, _c = self.decoderRNN(embed_input, state)[1]
        _h = _h.squeeze(0)
        dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
        
        e_t = self.v(torch.tanh(enc_proj + dec_proj).view(batch_size*max_enc_len, -1))
        attn_scores = e_t.view(batch_size, max_enc_len)
        del e_t
        attn_scores.masked_fill_(enc_mask, -float('inf'))
        attn_scores = F.softmax(attn_scores, 1)

        context = attn_scores.unsqueeze(1).bmm(enc_states)
        # output proj calculation
        p_vocab = F.softmax(self.V(torch.cat((_h, context.squeeze(1)), 1)), 1)
        # p_switch: [batch_size, 3]
        p_switch = torch.nn.functional.softmax(self.w_h(context.squeeze(1)) + self.w_s(_h) + self.w_x(embed_input[:, 0, :]) + self.w_c(enc_h_n), dim=1)
        p_switch = p_switch.view(-1, 3)
        
        # [batch_size, self.vocab_size]
        # general_words, pronouns, copying
        p_gen = torch.cat((p_switch[:, 0].view(-1, 1).expand(batch_size, self.vocab_size-self.nprons), p_switch[:, 1].view(-1, 1).expand(batch_size, self.nprons)), dim=1)
        assert p_gen.size(0) == batch_size and p_gen.size(1) == self.vocab_size
        p_copy = p_switch[:, 2].view(-1, 1) # [batch_size, 1]

        # weighted_Pvocab: [batch_size, vocab_sze]
        weighted_Pvocab = p_gen * p_vocab 
        weighted_attn = p_copy * attn_scores # [batch_size, max_enc_len]
        assert weighted_attn.size(0) == batch_size and weighted_attn.size(1) == max_enc_len

        if self.max_article_oov > 0:
            # create OOV (but in-article) zero vectors
            ext_vocab = torch.zeros((batch_size, self.max_article_oov), device=device, requires_grad=True)
            combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
            del ext_vocab
        else:
            combined_vocab = weighted_Pvocab
        assert article_inds.data.min() >=0 and article_inds.data.max() < (self.vocab_size+ self.max_article_oov), 'Recheck OOV indexes!'

        combined_vocab = combined_vocab.scatter_add(1, article_inds, weighted_attn)
        
        return combined_vocab, _h, _c.squeeze(0), p_switch.cpu().data.numpy()

    # Beam Search Decoding
    def decode(self, enc_states, enc_final_state, enc_mask, article_inds):
        """Parameters:
           enc_states [1, enc_seq_len, 2*hidden_size]: 
               Description encoder output states.
           enc_final_states ([1, hidden_size], ...):
               Context encoder final state
           enc_mask [1, enc_seq_len]:
               Description padding mask, 1 for pad token
           article_inds [1, enc_seq_len]: 
               Description encoder input with temporary OOV ids repalce 
               each OOV token
        """
        with torch.no_grad():
            assert enc_states.size(0) == enc_mask.size(0) == article_inds.size(0) == 1, "In decoding mode, the input batch size must be to 1"
            # _input: [batch_size(beam_size), seq_len]
            assert type(self.start_id) == int
            _input = torch.tensor([[self.start_id]], dtype=torch.long, device=device)
            init_state = enc_final_state[0].unsqueeze(0), enc_final_state[1].unsqueeze(0)
            enc_h_n = enc_final_state[0]
            decoded_outputs = []
            
            # all_hyps: list of current beam hypothesis. 
            all_hyps = [Hypothesis([self.start_id], None, None, 0)]
            # start decoding
            for _step in range(self.max_decode_steps):
                # ater first step, input is of batch_size=curr_beam_size
                # curr_beam_size <= self.beam_size due to pruning of beams that have terminated
                # adjust enc_states and init_state accordingly
                curr_beam_size  = _input.size(0)
                # [1, seq_len, 2*hidden] => [curr_beam_size, seq_len, 2*hidden]
                beam_enc_states = enc_states.expand(curr_beam_size, enc_states.size(1), enc_states.size(2)).contiguous().detach()
                # [1, enc_seq_len] => [curr_beam_size, enc_seq_len]
                beam_article_inds = article_inds.expand(curr_beam_size, article_inds.size(1)).detach()
                beam_enc_mask = enc_mask.expand(curr_beam_size, enc_mask.size(1)).detach()
                vocab_probs, next_h, next_c = self.decode_step(beam_enc_states, 
                    enc_h_n, init_state, _input, beam_enc_mask, beam_article_inds)

                # does bulk of the beam search
                # decoded_outputs: list of all ouputs terminated with stop tokens and of minimal length
                all_hyps, decode_inds, decoded_outputs, init_h, init_c = self.getOverallTopk(vocab_probs, next_h, next_c, all_hyps, decoded_outputs)

                # convert OOV words to unk tokens for lookup
                decode_inds.masked_fill_((decode_inds >= self.vocab_size), self.unk_id)
                decode_inds = decode_inds.t()
                _input = torch.tensor(decode_inds, device=device)
                init_state = (torch.tensor(init_h.unsqueeze(0), device=device), 
                              torch.tensor(init_c.unsqueeze(0), device=device))

            non_terminal_output = sorted(all_hyps, key=lambda x:x.survivability, reverse=True)
            sorted_outputs = sorted(decoded_outputs, key=lambda x:x.survivability, reverse=True)
            
            all_outputs = [item.full_prediction for item in sorted_outputs] + [item.full_prediction for item in non_terminal_output]
            return all_outputs

    def getOverallTopk(self, vocab_probs, _h, _c, all_hyps, results):
        """vocab_probs [curr_beam_size, vocab+oov_size]
        """
        # return top-k values i.e. top-k over all beams i.e. next step input ids
        # return hidden, cell states corresponding to topk
        probs, inds = vocab_probs.topk(k=self.beam_size, dim=1)
        probs = probs.log().data # [curr_beam_size, beam_size]
        inds = inds.data # [curr_beam_size, beam_size]

        candidates = []
        assert len(all_hyps) == probs.size(0), '# Hypothesis and log-prob size dont match'
        # cycle through all hypothesis in full beam
        for i, hypo in enumerate(probs.tolist()):
            for j, _ in enumerate(hypo):
                new_cand = all_hyps[i].extend(token_id=inds[i,j].item(),
                    hidden_state=_h[i].unsqueeze(0), cell_state=_c[i].unsqueeze(0), log_prob=probs[i,j])
                candidates.append(new_cand)
        # sort in descending order
        candidates = sorted(candidates, key=lambda x:x.survivability, reverse=True)
        new_beam, next_inp = [], []
        next_h, next_c = [], []
        # prune hypotheses and generate new beam
        for h in candidates:
            if h.full_prediction[-1] == self.stop_id:
                # weed out small sentences that likely have no meaning
                if len(h.full_prediction) >= self.min_length:
                    results.append(h)
            else:
                new_beam.append(h)
                next_inp.append(h.full_prediction[-1])
                next_h.append(h._h.data)
                next_c.append(h._c.data)
            if len(new_beam) >= self.beam_size:
                break
        assert len(new_beam) >= 1, 'Non-existent beam'
        return new_beam, torch.LongTensor([next_inp]), results, torch.cat(next_h, 0), torch.cat(next_c, 0)
    
    # greedy decoding
    def greedy_decoding(self, enc_states, enc_final_state, enc_mask, article_inds):
        """Parameters:
           enc_states [1, max_seq_len, hidden_size*2]:
               Output states for one description
           enc_final_state ([1, hidden_size], ...):
               Final state of one context(pre- and pos-).
           enc_mask [1, max_seq_len]:
               Padding mask for input description
           article_inds [1, max_seq_len]:
               Description as encoder input with temporary OOV ids repalce 
               each OOV token
        """
        with torch.no_grad():
            assert enc_states.size(0) == enc_mask.size(0) == article_inds.size(0) == 1, "In decoding mode, the input batch size must be to 1"
            _input = torch.tensor([[self.start_id]], dtype=torch.long, device=device)
            init_state = enc_final_state[0].unsqueeze(0), enc_final_state[1].unsqueeze(0)
            enc_h_n = enc_final_state[0]
            decode_outputs, switches = [self.start_id], [[0, 0, 0]]
            for _ in range(self.max_decode_steps):
                vocab_probs, next_h, next_c, switch_variable = self.decode_step(enc_states, 
                    enc_h_n, init_state, _input, enc_mask, article_inds)
                probs, inds = vocab_probs.topk(k=1) # probs: [1, 1]
                decode_outputs.append(inds.item())
                switches.append(switch_variable)
                
                if inds.item() == self.stop_id:
                    break
                
                assert inds.size(0) == inds.size(1) == 1
                if inds.max().item() >= self.vocab_size:
                    _input = torch.tensor([[self.unk_id]], device=device)
                else:
                    _input = inds.detach()
                init_state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
                
            return decode_outputs, switches


class REGModel(nn.Module):
    def __init__(self, word_dim, char_hidden_dim, hidden_size, vocab_size, wordEmbed, 
        start_id, stop_id, unk_id, nprons, beam_size=4, min_decode=3, max_decode=8, drop_out=0.5):
        super(REGModel, self).__init__()
        self.word_dim = word_dim
        self.char_hidden_dim = char_hidden_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.wordEmbed = wordEmbed
        
        self.start_id = start_id
        self.stop_id = stop_id
        self.unk_id = unk_id
        self.nprons = nprons
        self.beam_size = beam_size
        self.min_decode = min_decode
        self.max_decode = max_decode
        self.drop_out = drop_out
        
        self._build_model() # build model
    
    def _build_model(self):
        self.con_encoder = ContextEncoder(self.word_dim+self.char_hidden_dim, 
            self.hidden_size, self.wordEmbed, drop_out=self.drop_out)
        self.des_encoder = BidirectionalEncoder(self.word_dim+self.char_hidden_dim, self.hidden_size, self.wordEmbed, drop_out=self.drop_out)
        
        self.pointerDecoder = PointerAttentionDecoder(self.word_dim, 
            self.hidden_size, self.vocab_size, self.wordEmbed)
        self.pointerDecoder.setValues(self.start_id, self.stop_id, self.unk_id, self.nprons, self.beam_size, self.min_decode, self.max_decode)

    def forward(self, context_input, des_input, dec_input=None, decode_flag=False, beam_search=True):
        pre_context, pos_context = context_input
        des_word_ids, des_seq_lens, des_mask, des_char_ids, \
        des_word_lens, max_des_oovs, des_extented_ids = des_input
        
        # set num article OOVs in decoder
        self.pointerDecoder.max_article_oov = max_des_oovs
        self.pointerDecoder.beam_search = beam_search
        
        # encoding context
        h_n, c_n = self.con_encoder(pre_context, pos_context)
        # encoding description
        hidden_outs, _, _, mask = self.des_encoder(des_word_ids, des_seq_lens, 
            des_mask, des_char_ids, des_word_lens)
        
        if decode_flag:
            # decoding
            refex, switches = self.pointerDecoder(hidden_outs, (h_n, c_n), mask, 
                des_extented_ids, None, None, None, None, decode=True)
            return refex, switches
        else:
            assert dec_input is not None, "Description input can NOT be none in training model!"
            _input, target, dec_lens, dec_mask = dec_input
            total_loss = self.pointerDecoder(hidden_outs, (h_n, c_n), mask, 
                des_extented_ids, _input, target, dec_lens, dec_mask, decode=False)
            return total_loss


class REGShell(object):
    def __init__(self, config):
        super(REGShell, self).__init__()
        self.config = config
        self.logger = config.logger
        
        self._build_model() # build model
        
    def _build_model(self):
        self.wordEmbed = Embedding_layer(self.config.nwords, self.config.dim_word, self.config.nchars, 
            self.config.dim_char, self.config.hidden_size_char, self.config.embeddings, drop_out=self.config.drop_out).to(device)

        self.regModel = REGModel(self.config.dim_word, self.config.hidden_size_char, self.config.hidden_size, 
            self.config.nwords, self.wordEmbed, self.config.start_id, self.config.stop_id, self.config.unk_id, 
            self.config.npronouns, self.config.beam_size, self.config.min_dec_steps, self.config.max_dec_steps, 
            drop_out=self.config.drop_out)

        if torch.cuda.device_count() > 1:
            self.logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.regModel = nn.DataParallel(self.regModel)

        self.regModel.to(device)

        self.optimizer = torch.optim.Adam(self.regModel.parameters(), lr=self.config.lr)
    
    def save_model(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        
        save_path = self.config.dir_model + 'checkpoint.pth.tar'
        torch.save(self.regModel.state_dict(), save_path)
        self.logger.info("- model saved at: {}".format(save_path))
        
    def restore_model(self, dir_model):
        self.regModel.load_state_dict(torch.load(dir_model))
        self.logger.info("- model restored from: {}".format(dir_model))
        
    def train(self, train, dev, sample_set=None):
        """Performs training with early stopping and lr exponential decay
        """
        self.config.log_info()
        self.logger.info('start training...')
        best_score, nepoch_no_imprv = 0, 0 # for early stopping
        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1, 
                    self.config.nepochs))

            # shuffle the dataset
            if self.config.shuffle_dataset:
                train.shuffle()
            score = self.run_epoch(train, dev, epoch, samples=sample_set)
            self.config.lr *= self.config.lr_decay # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_model()
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    break
    
    def run_epoch(self, train, dev, epoch, samples=None):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better
            
        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, batch in enumerate(minibatches(train, batch_size)):
            # convert numpy data into torch tensor
            context_input, des_input, dec_input = self.data_prepare(batch)
            total_losses = self.regModel(context_input, des_input, dec_input, decode_flag=False)
            batch_loss = total_losses.mean()
            batch_loss.backward() # backward propagation

            # gradient clipping by norm
            if self.config.grad_clip > 0:
                clip_grad_norm_(self.regModel.parameters(), self.config.grad_clip)
            # update
            self.optimizer.step()
            self.optimizer.zero_grad()

            prog.update(i + 1, [("train loss",  batch_loss.detach())])

        # print out samples
        # self.predict(dataset)
        if samples is not None:
            self.logger.info('Evaluating samples...')
            pred_strings, all_contexts, all_des, all_refex, all_oovs, _ = self.predict(samples)
            for pres, contexts, des, refex, oovs in zip(pred_strings, all_contexts, all_des, all_refex, all_oovs):
                self.displayOutput(pres, contexts, des, refex, oovs)
        
        # evaluate the model
        self.logger.info('Evaluating development set...')
        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["acc"]

    def displayOutput(self, pres, context, description, refex, oovs, 
        show_ground_truth=True):
        if show_ground_truth:
            self.logger.info('- CONTEXT: {}'.format(context))
            self.logger.info('- DESCRIPTION: {}'.format(description))
            self.logger.info('- REFEX: {}'.format(refex))
        for i, pred in enumerate(pres):
            # [0, nwords - 1] [nwords, nwords+m-1]
            self.logger.info('- #{}: {}'.format((i+1), pred))
    
    def predict_batch(self, context_input, des_input, beam_search=True):
        """Predict referring expression on a batch of data

           Returns:
               preds: list of ids in greedy mode, list of list of ids in beam search mode
        """
        # self.regModel.eval() # set model to eval mode
        preds, switches = self.regModel(context_input, des_input, None, decode_flag=True, 
                              beam_search=beam_search)
        # self.regModel.train() # set model to train mode
        return preds, switches

    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset
        """
        self.logger.info("Testing model over test set...")
        if self.config.beam_search:
            self.logger.info("- beam searching")
        else:
            self.logger.info("- greedy decoding")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)
    
    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields Example object

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...
            
        """
        self.regModel.eval() # set model to eval mode
        assert self.wordEmbed.training == False
        assert self.regModel.training == False
        assert self.regModel.con_encoder.training == False
        # progbar stuff for logging
        batch_size = self.config.batch_size_eva
        nbatches = (len(test) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        total, correct = 0., 0.
        for i, batch in enumerate(minibatches(test, 1)):
            context_input, des_input, _ = self.data_prepare(batch)
            preds, _ = self.predict_batch(context_input, des_input, 
                beam_search=self.config.beam_search)
            target = batch.target_batch[0].tolist() # [1, max_dec_len]
            if self.config.beam_search:
                pred = preds[0] # in beam search mode, find the sequence with the highest probability
            else:
                pred = preds 

            stop_id = self.config.stop_id
            if stop_id in pred and stop_id in target:
                pred_trunc = pred[1: pred.index(stop_id) + 1] # get rid of start token
                target_trunc = target[: target.index(stop_id) + 1]

                if pred_trunc == target_trunc:
                    correct += 1
            elif pred == target:
                correct += 1
            total += 1
            # update progress bar
            prog.update(i + 1, [("acc", correct/total)])
        acc = correct/total
        self.regModel.train() # set model to train mode
        return {"acc": 100*acc}
    

    def predict(self, dataset):
        """Predict referring expression
        """
        self.regModel.eval() # set model to eval mode
        # progbar stuff for logging
        batch_size = self.config.batch_size_eva
        nbatches = (len(dataset) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        all_preds, all_contexts, all_des, all_refex, all_oovs = [], [], [], [], []
        all_switches = []
        if self.config.beam_search:
            self.logger.info("- beam searching")
        else:
            self.logger.info("- greedy decoding")
        for i, batch in enumerate(minibatches(dataset, 1)):
            context_input, des_input, _ = self.data_prepare(batch)
            preds, switches = self.predict_batch(context_input, des_input, beam_search=self.config.beam_search)
            # only select the most 3 possible beams
            if self.config.beam_search and len(preds) > 3:
                assert type(preds[0]) == list
                preds = preds[:3]
                
            all_switches.append(switches)
            all_preds.append(preds) # all_preds: [dataset_size, 3]
            all_contexts.append(batch.original_context[0]) # all_context: [dataset_size]
            all_des.append(batch.original_description[0])
            all_refex.append(batch.original_refex[0])
            all_oovs.append(batch.des_oovs[0])

            # update progress bar
            prog.update(i + 1)
        # convert predicted referring expression into strings
        pred_strings = []
        for i, (preds, oovs) in enumerate(zip(all_preds, all_oovs)):
            if type(preds[0]) == int:
                generated = ' '.join([self.config.id2word[ind] if ind < self.config.nwords else oovs[ind % self.config.nwords] for ind in preds])
                pred_strings.append([generated])
            else:
                beams = []
                for pred in preds:
                    generated = ' '.join([self.config.id2word[ind] if ind < self.config.nwords else oovs[ind % self.config.nwords] for ind in pred])
                    beams += [generated]
                pred_strings.append(beams)
        self.regModel.train() # set model to train mode
        return pred_strings, all_contexts, all_des, all_refex, all_oovs, all_switches
    
    def data_prepare(self, batch):
        """Convert numpy data to tensor for training
        """
        # context input
        prec_word_ids = torch.tensor(batch.prec_word_ids, device=device, dtype=torch.long)
        prec_seq_lens = torch.tensor(batch.prec_seq_lens, device=device, dtype=torch.long)
        prec_char_ids = torch.tensor(batch.prec_char_ids, device=device, dtype=torch.long)
        prec_word_lens = torch.tensor(batch.prec_word_lens, device=device, dtype=torch.long)

        posc_word_ids = torch.tensor(batch.posc_word_ids, device=device, dtype=torch.long)
        posc_seq_lens = torch.tensor(batch.posc_seq_lens, device=device, dtype=torch.long)
        posc_char_ids = torch.tensor(batch.posc_char_ids, device=device, dtype=torch.long)
        posc_word_lens = torch.tensor(batch.posc_word_lens, device=device, dtype=torch.long)    

        _pre = prec_word_ids, prec_seq_lens, prec_char_ids, prec_word_lens
        _pos = posc_word_ids, posc_seq_lens, posc_char_ids, posc_word_lens
        context_input = _pre, _pos

        # description input
        des_word_ids = torch.tensor(batch.des_word_ids, device=device, dtype=torch.long)
        des_seq_lens = torch.tensor(batch.des_seq_lens, device=device, dtype=torch.long)
        des_mask = torch.tensor(batch.des_mask, device=device, dtype=torch.long)
        des_char_ids = torch.tensor(batch.des_char_ids, device=device, dtype=torch.long)
        des_word_lens = torch.tensor(batch.des_word_lens, device=device, dtype=torch.long)
        max_des_oovs = torch.tensor(batch.max_des_oovs, device=device, dtype=torch.long)
        des_extented_ids = torch.tensor(batch.des_extented_ids, device=device, dtype=torch.long)
        des_input = des_word_ids, des_seq_lens, des_mask, des_char_ids, \
            des_word_lens, max_des_oovs, des_extented_ids
        
        # decoding input
        _input = torch.tensor(batch.dec_batch, device=device, dtype=torch.long)
        target = torch.tensor(batch.target_batch, device=device, dtype=torch.long)
        dec_lens = torch.tensor(batch.dec_lens, device=device, dtype=torch.long)
        dec_mask = torch.tensor(batch.dec_padding_mask, device=device, dtype=torch.long)
        dec_input = _input, target, dec_lens, dec_mask

        return context_input, des_input, dec_input
    
