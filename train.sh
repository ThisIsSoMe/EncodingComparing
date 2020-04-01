SEND_THREAD_NUM=1
# nohup python -u main_lstm.py --fine_tune >log_partial.txt &
# nohup python -u main_lstm.py --input_mode 'word_char' --output_dir 'saved5/' --fine_tune >log5.txt &
nohup python -u main_lstm.py --input_mode 'word_char' --model 'char_lstm_comparing' --output_dir 'saved9/' --neg_size 5 --fine_tune --gpu 2 --learning_rate 0.001 >log9.txt &