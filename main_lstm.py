from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import sys
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss, MarginRankingLoss
from argparse import ArgumentParser
logger = logging.getLogger(__name__)

## 
from torch.optim import Adam
# personal package
from modules.model import *
from args import get_args
from utils.corpus import *

def train_neg_sample(negs, neg_size, mode):
    new_negs = []
    if mode == 'word_char':
        for (words, chars) in negs:
            l = len(words)
            index = random.sample([i for i in range(l)], neg_size)
            new_negs.append((words[index], chars[index]))
    else:
        for item in negs:
            l = len(item)
            try:
                index = random.sample([i for i in range(l)], neg_size)
            except:
                from pdb import set_trace
                set_trace()
            new_negs.append(item[index])
    return new_negs

def cat_negs(data,mode='word'):
    # data: list n个[neg_size,cur_seq_len]
    neg_size = data[0].shape[0]
    max_seq_len = 0
    
    for item in data:
        seq_len = item.shape[1]
        max_seq_len = max(max_seq_len, seq_len)
    
    if mode == 'char':
        word_len = data[1].shape[-1]
        output = torch.LongTensor(len(data), neg_size, max_seq_len, word_len).fill_(0)
        for i, cur_tensor in enumerate(data):
            output[i,:,:cur_tensor.shape[1],:] = cur_tensor
    else:
        output = torch.LongTensor(len(data), neg_size, max_seq_len).fill_(0)
        for i, cur_tensor in enumerate(data):
            output[i,:,:cur_tensor.shape[-1]] = cur_tensor
    return output

def train_batchlize(data_tuple, batch_size, mode, shuffle=False):
    lens = len(data_tuple[-1])
    index = [i for i in range(lens)]
    if shuffle:
        index = random.shuffle(index)
    data_batches = []
    index_batches = []
    start = 0
    while start < lens-1:
        index_batches.append(index[start:start+batch_size])
        start += batch_size
    q, g, n = data_tuple
    if mode == 'word_char':
        # q:(word,char) g:(word,char) n:[(one_word,one_char)]
        for index_batch in index_batches:
            batch_q = (q[0][index_batch], q[1][index_batch])
            batch_g = (g[0][index_batch], g[1][index_batch])

            # 负样本batch化
            batch_neg_word = [n[i][0] for i in index_batch]
            batch_neg_char = [n[i][1] for i in index_batch]

            batch_neg_word = cat_negs(batch_neg_word, mode='word')

            batch_neg_char = cat_negs(batch_neg_char, mode='char')
            
            batch_n = (batch_neg_word, batch_neg_char)

            data_batches.append((batch_q, batch_g, batch_n))
    else:
        for index_batch in index_batches:
            cur_tensor = [n[i] for i in index_batch]
            cur_tensor = cat_negs(cur_tensor)
            data_batches.append((q[index_batch], g[index_batch], cur_tensor))
    return data_batches


def train(args):
    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # gpu
    if not args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print('使用%s号GPU' % args.gpu)

    # word vector
    corpus = Corpus()
    vocab, embed = corpus.load_embed(args.fn_embed)
    print("finish loading external word embeding, the shape is:")
    print(embed.shape)

    # model
    model_dict = {'lstm_comparing':BiLSTM_Encoding_Comparing, 'char_lstm_comparing':Char_BiLSTM_Encoding_Comparing}
    print("current model is", args.model)
    model_name = model_dict[args.model]
    if not args.no_cuda:
        embed = embed.cuda()
    model = model_name(args, embed)
    if not args.no_cuda:
        model.cuda()
    print(model)

    train_questions_raw, train_golds_raw, train_negs_raw = corpus.load_data(args.fn_train, 'train')
    valid_questions_raw, valid_golds_raw, valid_negs_raw = corpus.load_data(args.fn_valid, 'valid')

    train_questions = corpus.numericalize(train_questions_raw, args.input_mode)
    
    train_golds = corpus.numericalize(train_golds_raw, args.input_mode)
    train_negs = []
    for line in train_negs_raw:
        train_negs.append(corpus.numericalize(line, args.input_mode))
    
    # from pdb import set_trace
    # set_trace()
    if isinstance(train_questions, tuple):
        print("train data loaded!%d questions totally"%len(train_questions[0]))
    else:
        print("train data loaded!%d questions totally"%len(train_questions))

    valid_questions = corpus.numericalize(valid_questions_raw, args.input_mode)
    valid_golds = corpus.numericalize(valid_golds_raw, args.input_mode)
    valid_negs = []
    for index, line in enumerate(valid_negs_raw):
        valid_negs.append(corpus.numericalize(line, args.input_mode))
    
    if isinstance(valid_questions, tuple):
        print("valid data loaded!%d questions totally"%len(valid_questions[0]))
    else:
        print("valid data loaded!%d questions totally"%len(valid_questions))
    
    valid_dataset = (valid_questions, valid_golds, valid_negs)

    print("字符字典长度", corpus.len_char_dict())
    
    # dump vocab
    corpus.dump_vocab(args.vocab_word, mode='word')
    corpus.dump_vocab(args.vocab_char, mode='char')

    # training settings
    optimizer_dict = {"adam":Adam}
    optimizer_name = optimizer_dict[args.optimizer]
    print("choose optimizer:%s"%args.optimizer)
    optimizer = optimizer_name(model.parameters(), lr = args.learning_rate)
    
    criterion = MarginRankingLoss(margin=args.margin)
    
    patience = args.patience
    num_train_epochs = args.num_train_epochs
    iters_left = patience
    best_precision = 0
    num_not_improved = 0
    global_step = 0

    logger.info('\nstart training:%s'%datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("start training!")

    for epoch in range(args.num_train_epochs):
        # batchlize
        sample_train_negs = train_neg_sample(train_negs, args.neg_size, mode=args.input_mode)
        sample_train = (train_questions, train_golds, sample_train_negs)
        train_batches = train_batchlize(sample_train, args.batch_size, mode=args.input_mode)
        print("train data batchlized............")
        
        # 
        train_right = 0
        train_total = 0
        # 打印
        print('start time')
        start_time = datetime.now()
        logger.info('\nstart training:%s'%datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(start_time)

        model.train()
        optimizer.zero_grad()
        loss_epoch = 0 # 单次迭代的总loss
        for step, batch in enumerate(train_batches):
            # if not args.no_cuda:
            #     batch = (t.cuda() for t in batch)
            question_batch, gold_batch, negs_batch = batch
            pos_score, neg_scores = model(question_batch, gold_batch, negs_batch)
            
            pos_score = pos_score.expand_as(neg_scores).reshape(-1)
            neg_scores = neg_scores.reshape(-1)
            assert pos_score.shape == neg_scores.shape
            ones = torch.ones(pos_score.shape)
            if not args.no_cuda:
                ones = ones.cuda()
            loss = criterion(pos_score, neg_scores, ones)
            
            # evaluate train
            result = (torch.sum(pos_score.view(-1, args.neg_size) > neg_scores.view(-1, args.neg_size),-1) == args.neg_size)

            train_right += torch.sum(result).item()
            train_total += len(result)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss

        # 打印
        end_time = datetime.now()
        logger.info('\ntrain epoch %d time span:%s'%(epoch, end_time-start_time))
        print('train loss', loss_epoch.item())
        logger.info('train loss:%f'%loss_epoch.item())
        print('train result', train_right, train_total, 1.0*train_right/train_total)
        logger.info(('train result', train_right, train_total, 1.0*train_right/train_total))

        # eval
        right, total, precision = evaluate_char(args, model, valid_dataset)

        # print
        print('valid result', right, total, precision)
        print('epoch time')
        print(datetime.now())
        print('*'*20)
        logger.info("epoch:%d\t"%epoch+"dev_Accuracy-----------------------%d/%d=%f\n"%(right, total, precision))
        end_time = datetime.now()
        logger.info('dev epoch %d time span:%s'%(epoch,end_time-start_time))
        
        if precision > best_precision:
            best_precision = precision
            iters_left = patience
            print("epoch %d saved\n"%epoch)
            logger.info("epoch %d saved\n"%epoch)
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "best_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
        else:
            iters_left -= 1
            if iters_left == 0:
                break
    logger.info('finish training!')
    print('finish training!')


def evaluate_char(args, model, valid_dataset):
    model.eval()
    (valid_questions, valid_golds, valid_negs) = valid_dataset
    valid_questions = [(i,j) for i,j in zip(valid_questions[0], valid_questions[1])]
    valid_golds = [(i,j) for i,j in zip(valid_golds[0], valid_golds[1])]
    right, total = 0, 0
    for step,(q, gold, negs) in enumerate(zip(valid_questions, valid_golds, valid_negs)):
        question = (q[0].unsqueeze(0), q[1].unsqueeze(0))
        gold = (gold[0].unsqueeze(0), gold[1].unsqueeze(0))
        # if not args.no_cuda:
        #     question = question.cuda()
        #     gold = gold.cuda()

        # from pdb import set_trace
        # set_trace()
        gold_score = model.cal_score(question, gold).cpu()

        neg_scores = []
        batch_size = args.valid_batch_size
        if len(negs[0]) == 0:
            right += 1
        else:
            for index in range(0, len(negs), batch_size):
                negs_batch = (negs[0][index:index+batch_size], negs[1][index:index+batch_size])
                # if not args.no_cuda:
                #     negs_batch = negs_batch.cuda()
                
                # cur_scores = model.cal_score(question, negs_batch).cpu()
                try:
                    cur_scores = model.cal_score(question, negs_batch).cpu()
                except:
                    from pdb import set_trace
                    set_trace()
                neg_scores.extend([i for i in cur_scores])
            
            if sum([1 for i in neg_scores if gold_score>i]) == len(neg_scores):
                right += 1
        total += 1
        
    return right, total, 1.0*right/total


if __name__ == "__main__":
    
    # params
    args = get_args()
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print('maked dir %s' % args.output_dir)
    train(args)