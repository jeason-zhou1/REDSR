# CUDA_VISIBLE_DEVICES=0 python main.py --template EDSR --pre_train ../experiment/AIM_EDSR_512/model/model_best_2.pt --save_results --test_only --data_test AIM --save AIM_TEST_1
# CUDA_VISIBLE_DEVICES=0 python main.py --template EDSR --pre_train ../experiment/AIM_EDSR_512/model/model_best_ssim.pt --save_results --test_only --data_test AIM --save AIM_TEST_2
# CUDA_VISIBLE_DEVICES=0 python main.py --template EDSR --pre_train ../experiment/AIM_EDSR_SSIM_512/model/model_best.pt --save_results --test_only --data_test AIM --save AIM_TEST_3
# CUDA_VISIBLE_DEVICES=0 python main.py --template EDSR --pre_train ../experiment/AIM_EDSR_512/model/model_best.pt --save_results --test_only --data_test AIM --save AIM_TEST_4
# CUDA_VISIVLE_DEVICES=5 python main.py --template EDSR --pre_train ../experiment/AIM_EDSR_SSIM_512/model/model_best.pt --save_results --test_only --data_test Demo --dir_demo ../test/ValidationLR_x4 --save AIM_TEST_X4 --chop --self_ensemble


# CUDA_VISIBLE_DEVICES=5  python main.py --template EDSR --self_ensemble --pre_train ../experiment/AIM_EDSR_SSIM_512/model/model_best_2.pt --test_only --data_test Demo --dir_demo ../test/ValidationLR_x4 --save AIM_EDSR_FINAL_X4 --save_results --chop --chop_size 500
# CUDA_VISIBLE_DEVICES=5  python main.py --model DRLN --self_ensemble --pre_train ../../../pre_train/AIM_DRLN_X4.pt --test_only --data_test Demo --dir_demo ../test/ValidationLR_x4 --save AIM_DRLN_FINAL_X4 --save_results --chop --chop_size 500
# CUDA_VISIBLE_DEVICES=5  python main.py --template SAN --self_ensemble --pre_train ../experiment/AIM_SAN_X4/model/model_best_2.pt --test_only --data_test Demo --dir_demo ../test/ValidationLR_x4 --save AIM_SAN_FINAL_X4 --save_results --chop --chop_size 140 --shave_size 40
# python model_ensemble.py

CUDA_VISIBLE_DEVICES=3 python main.py --model NLSR_V4 --data_train DIV2KHF --save Final_X4_CARes --chop --chop_size 700 --save_results --batch_size 16 --loss 1*L1 --lr 1e-4 --decay 20-60-80-150-200 --test_every 1000  --n_GPUs 1 --scale 4 --patch_size 192 --data_test Set5+Set14+BSD100+Urban100 --pre_train ../experiment/Final_X4_CARes_hf/model/model_best.pt --self_ensemble --test_only --save_results
# CUDA_VISIBLE_DEVICES=5 python main.py --model NLSR_V4 --data_train DIV2KHF --save Final_X3_CARes_SSIM --chop --chop_size 700 --save_results --batch_size 16 --loss 1*CB+100*SSIM --lr 1e-4 --decay 20-60-80-150-200 --test_every 1000  --n_GPUs 1 --scale 3 --patch_size 144 --data_test Set5 --pre_train ../experiment/Final_X3_CARes_SSIM/model/model_best_2.pt 
# CUDA_VISIBLE_DEVICES=4  python main.py --model NLSR_V4 --data_train DIV2K --save RRAN_X4_testset --chop --chop_size 700 --save_results --batch_size 16 --loss 1*L1 --lr 1e-5 --decay 50-120-200 --test_every 1000  --n_GPUs 1 --scale 2 --patch_size 96 --data_test Set5+Set14+BSD100 --pre_train ../experiment/RRAN_X2_testset/model/model_best.pt 
# CUDA_VISIBLE_DEVICES=3  python main.py --model NLSR_V4 --data_train DIV2K --save RRAN_X3_testset --chop --chop_size 700 --save_results --batch_size 16 --loss 1*L1 --lr 1e-5 --decay 50-120-200 --test_every 1000  --n_GPUs 1 --scale 3 --patch_size 144 --data_test Set5+Set14+BSD100 --pre_train ../experiment/RRAN_X3_testset/model/model_best.pt 
# CUDA_VISIBLE_DEVICES=5  python main.py --template VDSR --data_train DIV2KHF --save VDSR --chop --chop_size 700 --save_results --batch_size 96 --loss 1*L1 --lr 1e-4 --decay 50-120-200  --n_GPUs 1 --scale 4 --patch_size 192 --data_test Set5

# CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py --model EDSR --data_train DIV2KHF  --save ff --chop --chop_size 700 --save_results --batch_size 32 --loss 1*L1 --lr 5e-5 --decay 50-100  --n_GPUs 2
# CUDA_VISIBLE_DEVICES=2 python main.py --model NLSR_V2 --data_train DIV2KHF --save RRAN+HF_test --chop --chop_size 700 --save_results --batch_size 32 --loss 1*L1+0.05*HF --lr 1.2e-5 --decay 80-150-200 --pre_train ../pre_train/RRAN.pt --n_GPUs 1 --cpu

# CUDA_VISIBLE_DEVICES=4  python main.py --template VDSR --data_train DIV2K --save VDSR  --n_GPUs 1 --scale 4 --patch_size 192 --data_test Set5+Set14+BSD100+Urban100 --pre_train ../experiment/VDSR/model/model_best.pt --test_only --save_results
