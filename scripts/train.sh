#!/usr/bin/env python
import os
### ViT-2D_8x32
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/vit_8x32_b16.yaml"
#os.system(cmd)

### TokShift_8x32
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/tokshift/tokshift_8x32_b16.yaml"
#os.system(cmd)

### TokShift_8x32_384
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/tokshift/tokshift_8x32_b16_384.yaml"
#os.system(cmd)

### TokShift_16x32_224
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/tokshift/tokshift_16x32_b16.yaml"
#os.system(cmd)

#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/tokshift/tokshift_12x32_l16.yaml"
#os.system(cmd)

### ViT_8x8
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/vit_8x8_b16.yaml"
#os.system(cmd)

### ViT_LAPS_8x8
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/vit_laps_8x8_b16.yaml"
#os.system(cmd)

### ViT_LAPS_8x8
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/vit_laps_8x8_b16_384.yaml"
#os.system(cmd)

### ViT_Large_LAPS_8x8
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/vit_large_laps_8x8_b16_384.yaml"
#os.system(cmd)

### Visformer_Base2D_8x8
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#        --tune_from pretrain/visformer_s_in10k.pth \
#		--cfg_file config/custom/k400/visformer/visformer_8x8.yaml"
#os.system(cmd)

### Visformer_Base3D_8x8
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#        --tune_from pretrain/visformer_s_in10k.pth \
#		--cfg_file config/custom/k400/visformer/visformer_8x8_base3D.yaml"
#os.system(cmd)

### Visformer_Plain_Shift_9x8
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#        --tune_from pretrain/visformer_s_in10k.pth \
#		--cfg_file config/custom/k400/visformer/visformer_plain_shift.yaml"
#os.system(cmd)

### Visformer_PS_8x8
cmd = "python -u main_tokshift.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23677 \
        --tune_from pretrain/visformer_s_in10k.pth \
		--cfg_file config/custom/k400/visformer/visformer_8x8_PS.yaml"
os.system(cmd)

### Visformer_LA_8x8
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#        --tune_from pretrain/visformer_s_in10k.pth \
#		--cfg_file config/custom/k400/visformer/visformer_8x8_LA.yaml"
#os.system(cmd)

### Visformer_LAPS_8x8
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#        --tune_from pretrain/visformer_s_in10k.pth \
#		--cfg_file config/custom/k400/visformer/visformer_LAPS_8x8.yaml"
#os.system(cmd)
