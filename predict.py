


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

# 删去实体中的带括号的描述信息
def del_des(string):
    stack=[]
    # if '_（' not in string and '）' not in string and '_(' not in string and ')' not in string:
    if '_' not in string:
        return string
    mystring=string[1:-1]
    if mystring[-1]!='）' and mystring[-1]!=')':
        return string
    for i in range(len(mystring)-1,-1,-1):
        char=mystring[i]
        if char=='）':
            stack.append('）')
        elif char == ')':
            stack.append(')')
        elif char=='（': 
            if stack[-1]=='）':
                stack=stack[:-1]
                if not stack:
                    break
        elif char=='(':
            if stack[-1]==')':
                stack=stack[:-1]
                if not stack:
                    break
    if mystring[i-1]=='_':
        i-=1
    else:
        return string
    return '<'+mystring[:i]+'>'

def predict_char(args, model):
    model.eval()

    # load data
    with open(args.input_file,'r')as f:
        data = json.load(f)
    with open(args.vocab_word,'r')as f:
        vocab_word = json.load(f)
    with open(args.vocab_char,'r')as f:
        vocab_char = json.load(f)
    corpus = Corpus()

    # 
    output_data = {}

    for line in data:
        q = line['q']
        paths = line['paths']

        q_input = [q]
        question = corpus.numericalize(q_input, mode=args.input_mode, words_dict=vocab_word, char_dict=vocab_char, state='test')
        paths_input = [''.join([del_des(i) for i in path]) for path in paths]
        paths_out = corpus.numericalize(paths_input, mode=args.input_mode, words_dict=vocab_word, char_dict=vocab_char, state='test')

        num_cands = len(paths_input)
        
        if num_cands == 0:
            print(q, "no path")
            continue

        cand_scores = []
        batch_size = args.valid_batch_size
        for index in range(0, num_cands, batch_size):
            cands_batch = (paths_out[0][index:index+batch_size], paths_out[1][index:index+batch_size])
            try:
                cur_scores = model.cal_score(question, cands_batch).cpu()
            except:
                from pdb import set_trace
                set_trace()
            cand_scores.extend([i.item() for i in cur_scores])

        cand_scores = torch.Tensor(cand_scores)
        
        # get topk answer
        if args.topk == 1:
            index = torch.argmax(cand_scores)
            output_data[q] = paths[index]
        else:
            _, index = torch.topk(cand_scores, k=min(args.topk,num_cands))
            output_data[q] = [paths[i] for i in index]
        
        fn_out = os.path.join(args.output_path,args.output_file)
        with open(fn_out, 'w')as f:
            json.dump(output_data, f, ensure_ascii=False)


def predict(args):
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
    model_state_dict = torch.load(args.model_path)
    if not args.no_cuda:
        embed = embed.cuda()
    model = model_name(args, embed)
    model.load_state_dict(model_state_dict)
    model.eval()

    if args.no_cuda == False:
        model.cuda()
    print('loaded model!')

    if args.input_mode == 'word_char':
        predict_char(args, model)

    # test_questions_raw, test_cands_raw = corpus.load_data(args.input_file, 'test')
    # test_questions = corpus.numericalize(test_questions_raw, args.input_mode)
    # test_cands = []
    # for index, line in enumerate(test_cands_raw):
    #     line = [''.join([del_des(i) for i in item]) for item in line]
    #     from pdb import set_trace
    #     set_trace()
    #     test_cands.append(corpus.numericalize(line, args.input_mode))
    # test_dataset = (test_questions, test_cands)


if __name__ == "__main__":
    # params
    args = get_args('predict')
    print(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print('maked dir %s' % args.output_path)
    predict(args)