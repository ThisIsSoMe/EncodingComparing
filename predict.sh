ONEHOP='one_hop_predict_path.json'
OUTDIR='data/char_lstm_comparing/'
MIXPATH=$OUTDIR'mix_paths.json'
FINALPATH='mix_predict_path.json'
# select one hop paths
# python predict.py --input_mode 'word_char' --model 'char_lstm_comparing' --output_dir 'saved5/' --fine_tune --no_cuda --topk 10 --output_path OUTDIR --output_file ONEHOP
# mix paths
python data/mix_paths.py --fn_in ONEHOP --fn_out MIXPATH
# predict final path
python predict.py --input_mode 'word_char' --model 'char_lstm_comparing' --output_dir 'saved5/' --fine_tune --no_cuda --topk 10 --input_file MIXPATH --output_path OUTDIR --output_file ONEHOP