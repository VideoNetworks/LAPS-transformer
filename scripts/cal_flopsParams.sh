#!/usr/bin/env python
import os
os.system('export CUDA_VISIBLE_DEVICES=0')

cmd = "python -u cal_flops.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23678 \
		--eval \
        --cfg_file config/custom/k400/visformer/visformer_LAPS_8x8.yaml"
        #--cfg_file config/custom/k400/visformer/visformer_8x8_LA.yaml"
        #--cfg_file config/custom/k400/visformer/visformer_8x8_PS.yaml"
        #--cfg_file config/custom/k400/visformer/visformer_plain_shift.yaml"
		#--cfg_file config/custom/k400/visformer/visformer_8x8_base3D.yaml"
		#--cfg_file config/custom/k400/visformer/visformer_8x8.yaml"
		#--cfg_file config/custom/k400/tokshift/tokshift_16x32_b16.yaml"
		#--cfg_file config/custom/k400/tokshift/tokshift_8x32_b16_384.yaml"
		#--cfg_file config/custom/k400/vit_8x32_b16.yaml"
os.system(cmd)

