#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from datautils import padding, padding_word_char
import torch.nn.functional as F
from torch import optim
import numpy as np
import random

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self, data):
        super(Model, self).__init__()
        self.input_size = data.input_size
        self.hidden_size = data.HP_hidden_dim
        self.output_size = data.label_alphabet_size
        self.vocal_size = data.word_alphabet_size
        self.embedding_size = data.HP_word_emb_dim
        self.CELoss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(data.HP_dropout)
        self.softmax = nn.LogSoftmax()
        self.use_cuda = data.HP_gpu
        self.embedding = nn.Embedding(self.vocal_size, self.embedding_size)
        self.dropout_rate = data.HP_dropout
        self.seed = data.HP_seed
        self.use_char = data.HP_use_char
        self.char_encoder = data.HP_char_features
        torch.manual_seed(self.seed)  # fixed the seed
        random.seed(self.seed)

        if data.pretrain_word_embedding is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(data.pretrain_word_embedding))

        if self.use_char:
            if self.char_encoder == "bilstm":
                self.char_feature = CharBiLSTM(self.use_cuda, data.char_alphabet_size, data.HP_char_emb_dim,
                                               data.HP_char_hidden_dim, data.HP_char_dropout,
                                               data.pretrain_char_embedding)
            elif self.char_encoder == "cnn":
                self.char_feature = CharCNN(self.use_cuda, data.char_alphabet_size, data.HP_char_emb_dim,
                                            data.HP_char_hidden_dim, data.HP_char_dropout, data.pretrain_char_embedding)


class LstmModel(Model):
    def __init__(self, data):
        super(LstmModel, self).__init__(data)

        self.linear = nn.Linear(self.hidden_size, self.output_size)

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            dropout=self.dropout_rate)

        self.w_i_in, self.w_i_on = self.lstm.all_weights[0][0].size()
        self.w_h_in, self.w_h_on = self.lstm.all_weights[0][1].size()
        self.lstm.all_weights[0][0] = Parameter(torch.randn(self.w_i_in, self.w_i_on)) * np.sqrt(2. / self.w_i_on)
        self.lstm.all_weights[0][1] = Parameter(torch.randn(self.w_h_in, self.w_h_on)) * np.sqrt(2. / self.w_h_on)

    def forward(self, input_x, input_char, input_y):
        """
        intput_x: b_s instances， 没有进行padding和Variable
        :param input:
        :return:
        """

        word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask = padding_word_char(
            self.use_cuda, input_x, input_char, input_y)

        input_x = word_seq_tensor
        input_y = label_seq_tensor

        embed_input_x = self.embedding(input_x)  # embed_intput_x: (b_s, m_l, em_s)

        batch_size = word_seq_tensor.size(0)
        sent_len = word_seq_tensor.size(1)

        if self.use_char:
            if self.use_cuda:
                char_features = self.char_feature.get_last_hiddens(char_seq_tensor,
                                                                   char_seq_lengths.numpy())
            else:
                char_features = self.char_feature.get_last_hiddens(char_seq_tensor,
                                                                   char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            embed_input_x = torch.cat([embed_input_x, char_features], 2)

        # embed_input_x = self.dropout(embed_input_x)

        # print 'inputs',embed_input_x.size()
        if self.use_cuda:
            embed_input_x_packed = pack_padded_sequence(embed_input_x, word_seq_lengths.cpu().numpy(), batch_first=True)
        else:
            embed_input_x_packed = pack_padded_sequence(embed_input_x, word_seq_lengths.numpy(), batch_first=True)

        encoder_outputs_packed, (h_last, c_last) = self.lstm(embed_input_x_packed)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)
        # print 'outputs',h_last.size()
        predict = self.linear(h_last)  # predict: [test.txt, b_s, o_s]
        #predict = self.softmax(predict.squeeze(0))  # predict.squeeze(0) [b_s, o_s]
        loss = self.CELoss(predict, input_y)

        if self.training:  # if it is in training module
            return loss
        else:
            value, index = torch.max(predict, 1)
            return index  # outsize, cal the acc


class BilstmModel(Model):
    def __init__(self, data):
        super(BilstmModel, self).__init__(data)

        self.linear = nn.Linear(self.hidden_size * 2, self.output_size)
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            dropout=self.dropout_rate,
                            bidirectional=True)

    def forward(self, input_x, input_char, input_y):
        """
        intput_x: b_s instances， 没有进行padding和Variable
        :param input_y:
        :param input_char:
        :param input_x:
        :return:
        """
        # input_x, batch_chars, input_y, sentence_lens, word_lens = padding(input_x, input_char, input_y)
        word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask = padding_word_char(
            self.use_cuda, input_x, input_char, input_y)

        input_x = word_seq_tensor
        input_y = label_seq_tensor

        embed_input_x = self.embedding(input_x)  # embed_intput_x: (b_s, m_l, em_s)

        batch_size = word_seq_tensor.size(0)
        sent_len = word_seq_tensor.size(1)

        if self.use_char:
            if self.use_cuda:
                char_features = self.char_feature.get_last_hiddens(char_seq_tensor,
                                                                   char_seq_lengths.numpy())
            else:
                char_features = self.char_feature.get_last_hiddens(char_seq_tensor,
                                                                   char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            embed_input_x = torch.cat([embed_input_x, char_features], 2)

        embed_input_x = self.dropout(embed_input_x)

        encoder_outputs, (h_last, c_last) = self.lstm(embed_input_x)
        h_last = torch.cat((h_last[0], h_last[1]), 1)

        predict = self.linear(h_last)  # predict: [test.txt, b_s, o_s]
        #predict = self.softmax(predict.squeeze(0))  # predict.squeeze(0) [b_s, o_s]

        loss = self.CELoss(predict, input_y)

        if self.training:  # if it is in training module
            return loss
        else:
            value, index = torch.max(predict, 1)
            return index  # outsize, cal the acc


class CnnModel(Model):

    def __init__(self, data):
        super(CnnModel, self).__init__(data)
        self.l2 = data.HP_l2
        self.kernel_size = [int(size) for size in data.HP_kernel_size.split("*")]
        self.kernel_num = [int(num) for num in data.HP_kernel_num.split("*")]
        nums = sum(self.kernel_num)
        self.linear = nn.Linear(nums, self.output_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num, (size, self.input_size)) for (size, num) in zip(self.kernel_size, self.kernel_num)])

    def forward(self, input_x, input_char, input_y):
        word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask = padding_word_char(
            self.use_cuda, input_x, input_char, input_y)

        input_x = word_seq_tensor
        input_y = label_seq_tensor
        batch_size = word_seq_tensor.size(0)
        sent_len = word_seq_tensor.size(1)

        self.poolings = nn.ModuleList([nn.MaxPool1d(sent_len - size + 1, 1) for size in
                                       self.kernel_size])  # the output of each pooling layer is a number

        input = input_x.squeeze(1)
        embed_input_x = self.embedding(input)  # embed_intput_x: (b_s, m_l, em_s)

        if self.use_char:
            if self.use_cuda:
                char_features = self.char_feature.get_last_hiddens(char_seq_tensor,
                                                                   char_seq_lengths.numpy())
            else:
                char_features = self.char_feature.get_last_hiddens(char_seq_tensor,
                                                                   char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            embed_input_x = torch.cat([embed_input_x, char_features], 2)

        embed_input_x = self.dropout(embed_input_x)
        embed_input_x = embed_input_x.view(embed_input_x.size(0), 1, -1, embed_input_x.size(2))

        parts = []  # example:[3,4,5] [100,100,100] the dims of data though pooling layer is 100 + 100 + 100 = 300
        for (conv, pooling) in zip(self.convs, self.poolings):
            conved_data = conv(embed_input_x).squeeze()
            if len(conved_data.size()) == 2:
                conved_data = conved_data.view(1, conved_data.size(0), conved_data.size(1))
            if len(conved_data.size()) == 1:
                conved_data = conved_data.view(1, conved_data.size(0), 1)
            pooled_data = pooling(conved_data).view(input_x.size(0), -1)
            parts.append(pooled_data)
        x = F.relu(torch.cat(parts, 1))

        # make sure the l2 norm of w less than l2
        w = torch.mul(self.linear.weight, self.linear.weight).sum().data[0]
        if w > self.l2 * self.l2:
            x = torch.mul(x.weight, np.math.sqrt(self.l2 * self.l2 * 1.0 / w))

        predict = self.linear(x)  # predict: [1, b_s, o_s]
        #predict = self.softmax(predict.squeeze(0))  # predict.squeeze(0) [b_s, o_s]

        loss = self.CELoss(predict, input_y)

        if self.training:  # if it is in training module
            return loss
        else:
            value, index = torch.max(predict, 1)
            return index  # outsize, cal the acc


class SumModel(Model):
    def __init__(self, data):
        super(SumModel, self).__init__(data)
        self.linear = nn.Linear(self.input_size, self.output_size)

    def forward(self, input_x, input_char, input_y):

        word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask = padding_word_char(
            self.use_cuda, input_x, input_char, input_y)

        input_x = word_seq_tensor
        input_y = label_seq_tensor

        embed_input_x = self.embedding(input_x)  # embed_intput_x: (b_s, m_l, em_s)

        batch_size = word_seq_tensor.size(0)
        sent_len = word_seq_tensor.size(1)

        if self.use_char:
            if self.use_cuda:
                char_features = self.char_feature.get_last_hiddens(char_seq_tensor,
                                                                   char_seq_lengths.numpy())
            else:
                char_features = self.char_feature.get_last_hiddens(char_seq_tensor,
                                                                   char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            embed_input_x = torch.cat([embed_input_x, char_features], 2)

        embed_input_x = self.dropout(embed_input_x)

        encoder_outputs = torch.zeros(len(input_y), self.input_size)  # 存放加和平均的句子表示

        if self.use_cuda:
            encoder_outputs = Variable(encoder_outputs).cuda()
        else:
            encoder_outputs = Variable(encoder_outputs)

        for index, batch in enumerate(embed_input_x):
            true_batch = batch[0:word_seq_lengths[index]]  # 根据每一个句子的实际长度取出实际batch
            encoder_outputs[index] = torch.mean(true_batch, 0)  # 平均

        predict = self.linear(encoder_outputs)
        #predict = self.softmax(predict)
        loss = self.CELoss(predict, input_y)

        if self.training:  # if it is in training module
            return loss
        else:
            value, index = torch.max(predict, 1)
            return index  # outsize, cal the acc

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(ChildSumTreeLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ioux = nn.Linear(self.input_size, 3 * self.hidden_size)
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.fx = nn.Linear(self.input_size, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)


    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
                self.fh(child_h) +
                self.fx(inputs).repeat(len(child_h), 1)
            )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, inputs):
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)

        return tree.state

class TreeLstm(Model):
    def __init__(self, data):
        super(TreeLstm,self).__init__(data)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.childSumTreeLstm = ChildSumTreeLSTM(self.input_size,self.hidden_size)

    def forward(self,tree,input_x,input_y):
        embed_input_x = self.embedding(input_x)
        state, hidden = self.childSumTreeLstm(tree,embed_input_x)

        predict = self.linear(hidden)
        #predict = self.softmax(predict)
        loss=0
        for inputy in input_y:
        	loss += self.CELoss(predict, inputy)
        if self.training:  # if it is in training module
            return loss
        else:
            pridicttensor=predict.data
            value, index=torch.topk(pridicttensor,5) #取出前五
            #value, index = torch.max(predict, 1)
            #print(index)
            return index  # outsize, cal the acc

class CharBiLSTM(nn.Module):
    def __init__(self, use_cuda, alphabet_size, embedding_dim, hidden_dim, dropout_rate, pretrined_embedding=None):
        super(CharBiLSTM, self).__init__()
        self.use_cuda = use_cuda
        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.char_drop = nn.Dropout(self.dropout_rate)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)

        if pretrined_embedding is not None:
            self.char_embeddings.weight = nn.Parameter(torch.FloatTensor(pretrined_embedding))

        self.char_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True,
                                 bidirectional=True)

        if self.use_cuda:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.char_lstm = self.char_lstm.cuda()

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        print batch_size
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        print char_rnn_out
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_rnn_out.transpose(1, 0)

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)


class CharCNN(nn.Module):
    def __init__(self, use_cuda, alphabet_size, embedding_dim, hidden_dim, dropout_rate, pretrined_embedding=None):
        super(CharCNN, self).__init__()
        print "build batched char cnn..."
        self.use_cuda = use_cuda
        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.char_drop = nn.Dropout(self.dropout_rate)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)

        if pretrined_embedding is not None:
            self.char_embeddings.weight = nn.Parameter(torch.FloatTensor(pretrined_embedding))

        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)

        if self.use_cuda:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.char_cnn = self.char_cnn.cuda()

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds).transpose(2, 1).contiguous()
        return char_cnn_out

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)


