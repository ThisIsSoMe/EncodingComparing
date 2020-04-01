import os
from argparse import ArgumentParser

def get_args(mode='train'):
    parser = ArgumentParser()
    ## Required parameters
    parser.add_argument("--model",
                        default='lstm_comparing',
                        type=str,
                        choices=['lstm_comparing','char_lstm_comparing'],
                        help='select model')
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help='seed for all random')
    parser.add_argument("--gpu",
                        default='3',
                        type=str,
                        help='choose gpu')
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help='do nots use cuda')
    parser.add_argument("--fn_embed",
                        default="data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5",
                        type=str,
                        help='external word embedding file')
    parser.add_argument("--fn_train",
                        default="../PathRanking/data/train.json",
                        type=str,
                        help='train data raw file')
    parser.add_argument("--fn_valid",
                        default="preprocess/valid_full_v2.json",
                        type=str,
                        help='valid data raw file')
    parser.add_argument("--output_dir",
                        default="saved/",
                        type=str,
                        help='output dir for saved model')
    parser.add_argument("--vocab_word",
                        default='data/vocab_word.txt',
                        type=str,
                        help="file for word vocabulary!")
    parser.add_argument("--vocab_char",
                        default='data/vocab_char.txt',
                        type=str,
                        help="file for char vocabulary!")
    

    ## Model parameters
    parser.add_argument("--hidden_size",
                        default=100,
                        type=int,
                        help='choose hidden size of rnn')
    parser.add_argument("--question_layers",
                        default=1,
                        type=int,
                        help='rnn layers of question encoder')
    parser.add_argument("--path_layers",
                        default=1,
                        type=int,
                        help='rnn layers of path encoder')
    
    ### char lstm
    parser.add_argument("--n_chars",
                        default=3500,
                        type=int,
                        help='lengths of char vocabulary')
    parser.add_argument("--dim_char",
                        default=100,
                        type=int,
                        help='dim of char embedding')
    parser.add_argument("--char_out",
                        default=50,
                        type=int,
                        help='lstm out dimension of char lstm')
    ### settings
    parser.add_argument("--fine_tune",
                        action='store_true',
                        help='fine tuning word embedding')
    parser.add_argument("--dropout_rate",
                        default=0.5,
                        type=float,
                        help='dropout prob')
    
    ## other parameters
    parser.add_argument("--input_mode",
                        default="word",
                        type=str,
                        choices=['word','char','word_char'],
                        help='input view')
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--valid_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--neg_size",
                        default=10,
                        type=int,
                        help="Size of negative sample.")
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--margin",
                        default=0.2,
                        type=float,
                        help="Margin for margin ranking loss.")
    parser.add_argument("--num_train_epochs",
                        default=100,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--patience",
                        default=10,
                        type=int,
                        help="Stop training when nums of epochs not improving.")
    parser.add_argument("--optimizer",
                        type=str,
                        default='adam',
                        choices=['adam'],
                        help="choose optimizer")
    ## while predicting
    if mode == 'predict':
         parser.add_argument("--model_path",
                        default='saved5/best_model.bin',
                        type=str,
                        help="the path of trained model!")
         parser.add_argument("--input_file",
                        default='data/one_hop_paths.json',
                        type=str,
                        help="the path of predict file!")
         parser.add_argument("--output_path",
                        default='data/output/',
                        type=str,
                        help="the folder for predicted file!")
         parser.add_argument("--output_file",
                        default='',
                        type=str,
                        help="the output file name after prediction")
         parser.add_argument("--test_batch_size",
                        default=64,
                        type=int,
                        help="batch size for test.")
         parser.add_argument("--topk",
                        default=1,
                        type=int,
                        help="topk paths while inferring.")
    args = parser.parse_args()
    return args