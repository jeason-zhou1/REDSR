# CUDA_VISIBLE_DEVICES=0 python main.py --template EDSR --pre_train ../experiment/AIM_EDSR_512/model/model_best_2.pt --save_results --test_only --data_test AIM --save AIM_TEST_1
# CUDA_VISIBLE_DEVICES=0 python main.py --template EDSR --pre_train ../experiment/AIM_EDSR_512/model/model_best_ssim.pt --save_results --test_only --data_test AIM --save AIM_TEST_2
# CUDA_VISIBLE_DEVICES=0 python main.py --template EDSR --pre_train ../experiment/AIM_EDSR_SSIM_512/model/model_best.pt --save_results --test_only --data_test AIM --save AIM_TEST_3
# CUDA_VISIBLE_DEVICES=0 python main.py --template EDSR --pre_train ../experiment/AIM_EDSR_512/model/model_best.pt --save_results --test_only --data_test AIM --save AIM_TEST_4
# CUDA_VISIVLE_DEVICES=5 python main.py --template EDSR --pre_train ../experiment/AIM_EDSR_SSIM_512/model/model_best.pt --save_results --test_only --data_test Demo --dir_demo ../test/ValidationLR_x4 --save AIM_TEST_X4 --chop --self_ensemble


# CUDA_VISIBLE_DEVICES=5  python main.py --template EDSR --self_ensemble --pre_train ../experiment/AIM_EDSR_SSIM_512/model/model_best_2.pt --test_only --data_test Demo --dir_demo ../test/ValidationLR_x4 --save AIM_EDSR_FINAL_X4 --save_results --chop --chop_size 500
# CUDA_VISIBLE_DEVICES=5  python main.py --model DRLN --self_ensemble --pre_train ../../../pre_train/AIM_DRLN_X4.pt --test_only --data_test Demo --dir_demo ../test/ValidationLR_x4 --save AIM_DRLN_FINAL_X4 --save_results --chop --chop_size 500
# CUDA_VISIBLE_DEVICES=5  python main.py --template SAN --self_ensemble --pre_train ../experiment/AIM_SAN_X4/model/model_best_2.pt --test_only --data_test Demo --dir_demo ../test/ValidationLR_x4 --save AIM_SAN_FINAL_X4 --save_results --chop --chop_size 140 --shave_size 40
python model_ensemble.py
