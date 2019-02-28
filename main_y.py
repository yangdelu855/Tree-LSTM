#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from tqdm import tqdm
from visdom import Visdom

from datautils import *
from module import *
import logging
import codecs
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,4'
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from MyDataset import MyDataset
from ModelFactory import ModelFactory
from utils import flatten
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from data import Data
from torch.utils.data import DataLoader
from treelstm_dataset import TreeDataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')


 
def tree_evaluate(ids,tree_path,model,data):
    dataset = TreeDataset(tree_path, ids)
    model.eval()
    f1,f2,f3,f4,f5,p1,p2,p3,p4,p5,r1,r2,r3,r4,r5=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    round_loss = 0
    predictions=torch.zeros(len(dataset))
    gold = torch.zeros(len(dataset))
    for idx in range(len(dataset)):
        tree, sent, label = dataset[idx]
        # print len(tree)
        # print sent
        # print label
        if data.HP_gpu:
            sent = Variable(torch.LongTensor(sent).cuda(),volatile=True)
            label = Variable(torch.LongTensor(label).cuda(),volatile=True)
        else:
            sent = Variable(torch.LongTensor(sent),volatile=True)
            label = Variable(torch.LongTensor(label),volatile=True)

        correct = model(tree, sent, label)
        # 看一下这里是什么格式的
        correctcpu=correct.cpu()
        reallabelcpu=label.cpu().data
        #print(reallabelcpu)
        correctlabel=[]
        correctlabel.append(correctcpu[0][0])
        correctlabel.append(correctcpu[0][1])
        correctlabel.append(correctcpu[0][2])
        correctlabel.append(correctcpu[0][3])
        correctlabel.append(correctcpu[0][4])
        reallabel=[]
        for i in range(0,len(reallabelcpu)):
               if reallabelcpu[i] not in reallabel:
        	      reallabel.append(reallabelcpu[i])
        #print(reallabel)
        #print(correctlabel)
        f1_score1,pp1,rr1=compute_f1_score(correctlabel[:1],reallabel)
        f1_score2,pp2,rr2=compute_f1_score(correctlabel[:2],reallabel)
        f1_score3,pp3,rr3=compute_f1_score(correctlabel[:3],reallabel)
        f1_score4,pp4,rr4=compute_f1_score(correctlabel[:4],reallabel)
        f1_score5,pp5,rr5=compute_f1_score(correctlabel,reallabel)
        f1,p1,r1=f1+f1_score1,p1+pp1,r1+rr1
        f2,p2,r2=f2+f1_score2,p2+pp2,r2+rr2
        f3,p3,r3=f3+f1_score3,p3+pp3,r3+rr3
        f4,p4,r4=f4+f1_score4,p4+pp4,r4+rr4
        f5,p5,r5=f5+f1_score5,p5+pp5,r5+rr5
    f1,p1,r1=f1/len(dataset),p1/len(dataset),r1/len(dataset)
    f2,p2,r2=f2/len(dataset),p2/len(dataset),r2/len(dataset)
    f3,p3,r3=f3/len(dataset),p3/len(dataset),r3/len(dataset)
    f4,p4,r4=f4/len(dataset),p4/len(dataset),r4/len(dataset)
    f5,p5,r5=f5/len(dataset),p5/len(dataset),r5/len(dataset)
    print("------------------------------------------------------------")
    print("Top1F1 Score:%.3f \t Precision:%.3f \t Recall:%.3f" % (f1,p1,r1))
    print("Top2F1 Score:%.3f \t Precision:%.3f \t Recall:%.3f" % (f2,p2,r2))
    print("Top3F1 Score:%.3f \t Precision:%.3f \t Recall:%.3f" % (f3,p3,r3))
    print("Top4F1 Score:%.3f \t Precision:%.3f \t Recall:%.3f" % (f4,p4,r4))
    print("Top5F1 Score:%.3f \t Precision:%.3f \t Recall:%.3f" % (f5,p5,r5))
    print("------------------------------------------------------------")
    return f1

def compute_f1_score(label_list_top5,eval_y_short):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    num_correct_label=0.0
    for label_predict in label_list_top5:
        if label_predict in eval_y_short:
            num_correct_label=num_correct_label+1.0
    #P@5=Precision@5
    num_labels_predicted=len(label_list_top5)
    all_real_labels=len(eval_y_short)
    p_5=num_correct_label/(num_labels_predicted*1.0)
    #R@5=Recall@5
    r_5=num_correct_label/(all_real_labels*1.0)
    if(p_5!=0 and r_5!=0):
        f1_score=2.0*p_5*r_5/(p_5+r_5)
    else:
        f1_score=0.0
    return f1_score,p_5,r_5


def evaluate(ids, model, batch_size):
    '''

    :param instance_x:
    :param instance_y:
    :param model:
    :return:
    '''
    dataset = MyDataset(ids)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    # start_id, end_id = 0, 0
    model.eval()
    # lst = list(range(len(instance_x)))
    gold_all, predict_all = [], []

    for step, (batch_words, batch_chars, batch_label) in enumerate(dataloader):
        model.eval()
        predict = model.forward(batch_words, batch_chars, batch_label)  # 进到forward的时候，顺序是变了,降序排列了

        lst = range(len(batch_words))
        lst = sorted(lst, key=lambda d: -len(batch_words[d]))
        batch_instance_y = [batch_label[index] for index in lst]  # sorted by descend

        predict_all.append(predict.data.tolist())

        gold_all.append(batch_instance_y)

    gold = flatten(gold_all)
    predict = flatten(predict_all)

    sum_all = len(gold)
    correct = map(cmp, gold, predict).count(0)

    return correct * 1.0 / sum_all


def main():
    cmd = argparse.ArgumentParser("sentence_representation_library")
    cmd.add_argument("--train", help='train data_path', type=str, default='../debugdata/newtrain-mul.toks')
    cmd.add_argument("--dev", help='dev data_path', type=str, default='../debugdata/newdev-mul.toks')
    cmd.add_argument("--test", help='test data_path', type=str, default='../debugdata/newtest-mul.toks')
    # cmd.add_argument("--train", help='train data_path', type=str, default='/users4/bbcai/data/amazon/books/1600_train.review.tok')
    # cmd.add_argument("--dev", help='dev data_path', type=str, default='/users4/bbcai/data/amazon/books/400_test.review.tok')
    # cmd.add_argument("--test", help='test data_path', type=str, default='/users4/bbcai/data/amazon/books/400_test.review.tok')
    cmd.add_argument("--train_tree", help='train tree data_path',type=str,default='../debugdata/train/newa.parents')
    cmd.add_argument("--dev_tree", help='dev tree data_path',type=str,default='../debugdata/dev/newa.parents')
    cmd.add_argument("--test_tree", help='test tree data_path',type=str,default='../debugdata/test/newa.parents')
    cmd.add_argument("--number_normalized", help='number_normalized', action="store_true")
    cmd.add_argument("--batch_size", help='batch_size', type=int, default=10)
    cmd.add_argument("--max_epoch", help='max_epoch', type=int, default=6)
    cmd.add_argument("--hidden_size", help='hidden_size', type=int, default=200)
    cmd.add_argument("--embedding_size", help='embedding_size', type=int, default=300)
    cmd.add_argument("--embedding_path", default="../data/glove.840B.300d.txt", help="pre-trained embedding path") #
    cmd.add_argument("--lr", help='lr', type=float, default=0.01)
    cmd.add_argument("--seed", help='seed', type=int, default=1)
    cmd.add_argument("--dropout", help="dropout", type=float, default=0.2)
    cmd.add_argument("--kernel_size",
                     help="kernel_size[Attention:the kernal size should be smaller than the length of your input after padding]",
                     type=str, default="3*4*5")
    cmd.add_argument("--kernel_num", help="kernel_num", type=str, default="100*100*100")
    cmd.add_argument("--l2", help="l2 norm", type=int, default=3)
    cmd.add_argument("--encoder", help="options:[lstm, bilstm, gru, cnn, treelstm, sum]", type=str, default='treelstm')
    cmd.add_argument("--gpu", action="store_true", help="use gpu",default =True)
    cmd.add_argument("--model_name", default="sr", help="model name")
    cmd.add_argument("--optim", default="Adam", help="options:[Adam,SGD]")
    cmd.add_argument("--load_model", default="", help="model path")
    # character
    cmd.add_argument("--char_encoder", help="options:[bilstm, cnn]", type=str, default='')
    cmd.add_argument("--char_hidden_dim", help="char_hidden_dim", type=int, default=50)
    cmd.add_argument("--char_embedding_path", help='char_embedding_path', default="")
    cmd.add_argument("--char_embedding_size", help='char_embedding_size', type=int, default=50)
    cmd.add_argument("--char_dropout", help="char_dropout", type=float, default=0.1)

    args = cmd.parse_args()
    # fixed the seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data = Data(args)
    data.HP_gpu=True
    # build word character label alphabet
    data.build_alphabet(args.train)

    data.build_alphabet(args.dev)
    data.build_alphabet(args.test)
    data.fix_alphabet()

    # prepare data
    data.generate_instance(args.train, 'train')
    data.generate_instance(args.dev, 'dev')
    data.generate_instance(args.test, 'test')

    # load pre-trained embedding(if not,we random init the embedding using nn.Embedding())
    if args.embedding_path:
        data.build_word_pretrain_emb(args.embedding_path)
    if args.char_embedding_path:
        data.build_char_pretrain_emb(args.char_embedding_path)

    # create visdom enviroment
    #vis = Visdom(env=data.HP_model_name)
    # check visdom connection
    #vis_use = vis.check_connection()

    if data.HP_use_char:
        if data.HP_char_features == "bilstm":
            data.input_size = data.HP_word_emb_dim + 2 * data.HP_char_hidden_dim
        elif data.HP_char_features == "cnn":
            data.input_size = data.HP_word_emb_dim + data.HP_char_hidden_dim
    else:
        data.input_size = data.HP_word_emb_dim

    # create factory and type create the model according to the encoder
    factory = ModelFactory()
    model = factory.get_model(data)

    # load model
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))

    if data.HP_gpu:
        model = model.cuda()

    # Dataset、DataLoader for Batch
    if data.HP_encoder_type == 'treelstm':
        dataset = TreeDataset(args.train_tree, data.train_Ids)
    else:
        dataset = MyDataset(data.train_Ids)
        dataloader = DataLoader(dataset=dataset, batch_size=data.HP_batch_size, shuffle=True, collate_fn=collate_batch)

    best_valid_acc = 0.0

    # optimizer
    if data.HP_optim.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr)
    elif data.HP_optim.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr)

    # print dataset[0]
    model.train()
    data.show_data_summary()  # show information about the hyper parameters and some datas
    for epoch in range(data.HP_iteration):
        round_loss = 0
        logging.info("epoch:{0} begins!".format(epoch))
        if data.HP_encoder_type == 'treelstm':
            optimizer.zero_grad()
            total_loss = 0.0
            indices = torch.randperm(len(dataset))
            model.train()
            for idx in range(len(dataset)):
                tree, sent, label = dataset[indices[idx]]
                if data.HP_gpu:
                    sent = Variable(torch.LongTensor(sent).cuda())
                    label = Variable(torch.LongTensor(np.array(label,dtype=np.int64)).cuda())
                else:
                    sent = Variable(torch.LongTensor(sent))
                    label = Variable(torch.LongTensor(np.array(label,dtype=np.int64)))
                loss = model(tree, sent, label)
                loss.backward()
                round_loss += loss.data[0]
                if idx % data.HP_batch_size == 0 and idx > 0:
                    optimizer.step()
                    optimizer.zero_grad()
        else:
            for step, (batch_words, batch_chars, batch_label) in enumerate(dataloader):
                model.train()
                optimizer.zero_grad()  # zero grad
                loss = model.forward(batch_words, batch_chars, batch_label)
                loss.backward()  # back propagation
                optimizer.step()  # update parameters
                round_loss += loss.data[0]  # the sum of the each epoch`s loss
        logging.info("epoch:{0} loss:{1}".format(epoch, round_loss))

        # draw loss
        #if vis_use:
            #vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor(round_loss), win='loss',
                     #update='append' if epoch > 0 else None)

        # use current model to test on the dev set

        if data.HP_encoder_type == 'treelstm':
            valid_acc = tree_evaluate(data.dev_Ids,args.dev_tree,model,data)
        else:
            valid_acc = evaluate(data.dev_Ids, model, data.HP_batch_size)
        logging.info("valid_acc = {0}".format(valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            # test on the test set
            if data.HP_encoder_type == 'treelstm':
                test_acc = tree_evaluate(data.test_Ids,args.test_tree,model,data)
            else:
                test_acc = evaluate(data.test_Ids, model, data.HP_batch_size)

            # draw test acc
            #if vis_use:
                #vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([test_acc]), win='test_acc',
                         #update='append' if epoch > 0 else None)

            # save model
            torch.save(model.state_dict(), "../model/" + data.HP_model_name + ".model")
            logging.info(
                "epoch:{0} New Record! valid_accuracy:{1}, test_accuracy:{2}".format(epoch, valid_acc, test_acc))

    # finally, we evaluate valid and test dataset accuracy
    if data.HP_encoder_type == 'treelstm':
        valid_acc = tree_evaluate(data.dev_Ids,args.dev_tree,model,data)
        test_acc = tree_evaluate(data.test_Ids,args.test_tree,model,data)
    else:
        valid_acc = evaluate(data.dev_Ids, model, data.HP_batch_size)
        test_acc = evaluate(data.test_Ids, model, data.HP_batch_size)

    logging.info("Train finished! saved model valid acc:{0}, test acc: {1}".format(valid_acc, test_acc))


if __name__ == '__main__':
    main()
