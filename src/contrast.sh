CUDA_VISIBLE_DEVICES=4 python main.py --data_train DIV2KHF --model RCA_16 --save contrast_16 --batch_size 8 --lr 1e-4 --decay 20-60-80-150-200 --n_GPUs 1 --data_test Set5+Set14
