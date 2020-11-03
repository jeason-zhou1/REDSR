# EDSR baseline model (x2) + JPEG augmentation
python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]
<<<<<<< HEAD
=======
CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --reset --pre_train ../experiment/edsr_x2/model/model_best.pt --n_GPUs 1 --batch_size 24
>>>>>>> 2e1610cc3343ea8e249d1205582ca21fb7ea3db6

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+BSD100+Urban100+DIV2K --data_range 1-100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble
<<<<<<< HEAD
python main.py --template RCAN --save RCAN_test --scale 4 --save_results --pre_train /mnt/train_SR/EDSR-train/experiment/RCAN_BIX4_G10R20P48/model/model_best.pt --data_test Set5+Set14+BSD100+Urban100+DIV2K --data_range 1-100 --test_only
# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results
CUDA_VISIBLE_DEVICES=6 python main.py --data_test Demo --scale 4 --test_only --save_results --dir_demo /mnt/train_SR/EDSR-Pytorch-master/test/Validation_x4 
=======

# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results

>>>>>>> 2e1610cc3343ea8e249d1205582ca21fb7ea3db6
# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt

<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 4 --save AIM_edsr_x4_finetune --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_train AIM --data_test AIM --batch_size 24
CUDA_VISIBLE_DEVICES=4 python main.py --model EDSR --scale 3 --save AIM_edsr_x3_finetune --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_train AIM --data_test AIM --batch_size 24
CUDA_VISIBLE_DEVICES=5 python main.py --model EDSR --scale 2 --save AIM_edsr_x2_finetune --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_train AIM --data_test AIM --batch_size 24
CUDA_VISIBLE_DEVICES=4 python main.py --model EDSR --scale 2 --save AIM_edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --data_train AIM --data_test AIM --batch_size 24
CUDA_VISIBLE_DEVICES=5 python main.py --model SAN --save SAN_BIX4_G20R10P48 --scale 4 --n_resgroups 20 --n_resblocks 10 --n_feats 64  --reset --chop --save_results --patch_size 192 --data_train AIM --data_test AIM --batch_size 16 
python main.py --data_test Set5+Set14+BSD100 --pre_train /mnt/train_SR/BasicSR-master/pretrained_models/RRDB_PSNR_x4.pth --test_only --model RRDB --save ESRGAN --save_results --rgb_range 1
=======
CUDA_VISIBLE_DEVICES=5 python main.py --model SAN --save SAN_BIX4_G20R10P48 --scale 4 --n_resgroups 20 --n_resblocks 10 --n_feats 64  --reset --chop --save_results --patch_size 192  --data_train AIM --data_test AIM
>>>>>>> 2e1610cc3343ea8e249d1205582ca21fb7ea3db6
