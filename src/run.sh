CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py --template EDSR --self_ensemble --pre_train ../pre_train/AIM_EDSR_X4.pt --test_only --data_test Demo --dir_demo ../testLR_x4 --save AIM_EDSR_FINAL_X4 --save_results --chop --chop_size 400 --n_GPUs 1
CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py --model DRLN --self_ensemble --pre_train ../pre_train/AIM_DRLN_X4.pt --test_only --data_test Demo --dir_demo ../testLR_x4 --save AIM_DRLN_FINAL_X4 --save_results --chop --chop_size 400 200 --shave_size 40 20 --n_GPUs 1

python model_ensemble.py
