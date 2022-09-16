#!/usr/bin/env python
import os
cmd = "python -u main_tokshift.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23677 \
		--eval \
        --resume video_checkpoints/Visformer_LAPS_visformer_small_kinetics_C400_8x8_E18_LR0.09023_B40_S224_D4_SLevel_s2_1_2_3_1_s3_2_3_1_2/best_ckpt_e15.pth \
        --cfg_file config/custom/k400/visformer/visformer_LAPS_8x8.yaml"
        #--resume video_checkpoints/TokShift_vit_base_patch16_224_in21k_kinetics_C400_8x32_E18_LR0.23_B24_S224_D4/tokshift_8x32_224_e17.pth \
		#--cfg_file config/custom/k400/tokshift/tokshift_8x32_b16.yaml"
os.system(cmd)
