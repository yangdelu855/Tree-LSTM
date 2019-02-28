#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


def collate_batch(batch):
    outputs_words = []
    outputs_chars = []
    outputs_lables = []
    for key in batch:
        outputs_words.append(key[0])
        outputs_chars.append(key[1])
        for label in key[2]:
            outputs_lables.append(label)
    return outputs_words, outputs_chars, outputs_lables


def padding(instance_x, batch_chars, instance_y):
    lst = range(len(instance_x))
    # 按照长度排序
    lst = sorted(lst, key=lambda d: -len(instance_x[d]))
    # 重新排序过后的
    instance_x_sorted = [instance_x[index] for index in lst]  # be sorted in decreasing order for packed
    instance_y_sorted = [instance_y[index] for index in lst]
    # 记录padding之前的长度
    sentence_lens = [len(sentence) for sentence in instance_x_sorted]  # for pack-padded deal
    max_len = max(sentence_lens)
    # 根据词典，使用1来进行padding
    instance_x_sorted = [sentence + (max_len - len(sentence)) * [0] for sentence in instance_x_sorted]

    # 首先根据句子长度进行交换
    batch_chars_sorted = [batch_chars[index] for index in lst]

    words_lens = []
    character_padding_res = []
    # 对character级别进行padding
    for index, sentence in enumerate(batch_chars_sorted):
        # print sentence
        c_lst = range(len(sentence))
        # print c_lst
        c_lst = sorted(c_lst, key=lambda d: -len(sentence[d]))
        # print c_lst
        sentence_sorted = [sentence[index] for index in c_lst]
        # print sentence_sorted
        words_len = [len(word) for word in sentence_sorted]
        # print words_len
        words_lens.append(words_len)
        # print words_lens
        max_word_len = max(words_len)
        # print max_word_len
        sentence_sorted = [word + (max_word_len - len(word)) * [0] for word in sentence_sorted]
        # print sentence_sorted
        character_padding_res.append(sentence_sorted)

    return instance_x_sorted, character_padding_res, instance_y_sorted, sentence_lens, words_lens


def padding_word_char(use_cuda, batch_words, batch_chars, batch_label):
    batch_size = len(batch_words)

    words = batch_words
    chars = batch_chars
    labels = batch_label

    word_seq_lengths = torch.LongTensor(map(len, words))  # 每一句话句子长度 [torch.LongTensor of size 2]
    max_seq_len = word_seq_lengths.max()  # 句子的最大长度 number
    word_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len))).long()
    mask = Variable(torch.zeros((batch_size, max_seq_len))).byte()

    # 用0对word进行padding
    for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)  # 按照长度降序排序
    word_seq_tensor = word_seq_tensor[word_perm_idx]  # 得到的tensor也对应排序
    mask = mask[word_perm_idx]

    label_seq_tensor = Variable(torch.LongTensor(labels))
    label_seq_tensor = label_seq_tensor[word_perm_idx]  # 转换label与之前降序排序之后的对应

    # character padding
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]  # 首选补全word，使得长度一样
    length_list = [map(len, pad_char) for pad_char in pad_chars]  # 得到每句话每个词的长度（上面补充的0也算）
    max_word_len = max(map(max, length_list))  # 取出所有中长度最长的
    char_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len, max_word_len))).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)  # 按照上面的顺序调换
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]  # 已经被完全打乱，完全按照word长度来进行排序

    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    if use_cuda:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()

    return word_seq_tensor, word_seq_lengths, word_seq_recover, \
           char_seq_tensor, char_seq_lengths, char_seq_recover, \
           label_seq_tensor, \
           mask


def normalize_word(word):
    """
    讲英语单词中的数字全部变为0
    :param word:
    :return:
    """
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, char_alphabet, label_alphabet, number_normalized):
    in_lines = open(input_file, 'r').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    word_Ids = []
    char_Ids = []
    label_Ids = []
    for line in in_lines:
        line = line.strip()
        if line:
            pairs = line.strip().split('|||')
            label = pairs[0].strip().split(' ')
            labels.append(label)
            wwww=pairs[1].strip().split(' ')
            for la in label:
            	label_Ids.append(label_alphabet.get_index(la))
            for word in wwww:
                if number_normalized:
                    word = normalize_word(word)
                words.append(word)
                word_Ids.append(word_alphabet.get_index(word))
                char_list = []
                char_Id = []
                for char in word:
                    char_list.append(char)
                for char in char_list:
                    char_Id.append(char_alphabet.get_index(char))
                chars.append(char_list)
                char_Ids.append(char_Id)
            instence_texts.append([words, chars, labels])
            instence_Ids.append([word_Ids, char_Ids, label_Ids])
            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            # 除去最开始的单词
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0].decode('utf-8')] = embedd
    return embedd_dict, embedd_dim


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    """
    构建预训练向量
    :param embedding_path:
    :param word_alphabet:
    :param embedd_dim:
    :param norm:
    :return:
    """
    embedd_dict = dict()

    # 加载文件中有的预训练向量
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0

    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    return pretrain_emb, embedd_dim
