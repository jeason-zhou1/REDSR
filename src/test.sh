CUDA_VISIBLE_DEVICES=6 python main.py --test_only --model NLSR_V2 --pre_train ../experiment/RRAN+0.05HF/model/model_best.pt --data_test Set5+Set14+BSD100+Urban100 --chop --chop_size 700 --save TEST_RRAN 