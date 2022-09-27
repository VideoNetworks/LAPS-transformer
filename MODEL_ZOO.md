# Model Zoo and Baselines
We retrain all models with Version2.0, the value fluctuates normally without affecting conclusion. We change FLOPs measurement to fvcore tool, thereby slightly different from [`TokShift`](https://github.com/VideoNetworks/TokShift-Transformer), Pretrained-Weights [`Visformer_s_in22k`](https://drive.google.com/file/d/1uPIMLDcMgMmj-Fp5q-cibIuf1Nebdhxu/view?usp=sharing)
## LAPS on Kinetcis-400 val
| Arch | Backbone |  Pretrain |  Res & Frames & Step| GFLOPs* x views| Params | Ver2.0 (top-1 Accuracy) | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ---------------------- | ------------- | ------------- |
| Base2D | Visf | IN10k | 224 & 8 & 8| 39.074 x 30 | 39.758 | - |  [`visformer_8x8.yaml`](config/custom/k400/visformer/visformer_8x8.yaml) |
| Base3D | Visf | IN10k | 224 & 8 & 8| 46.509 x 30 | 39.758 | - |  [`visformer_8x8_base3D.yaml`](config/custom/k400/visformer/visformer_8x8_base3D.yaml) |
| PlainShift | Visf | IN10k | 224 & 8 & 8| 39.074 x 30 | 39.758 | - | [`visformer_plain_shift.yaml`](config/custom/k400/visformer/visformer_plain_shift.yaml) |
| PeriodicShift (PS) | Visf | IN10k | 224 & 8 & 8| 39.074 x 30 | 39.758 | 75.48 | [`visformer_8x8_PS.yaml`](config/custom/k400/visformer/visformer_8x8_PS.yaml) [`pth`]()|
| LeapAttention (LA)| Visf | IN10k | 224 & 8 & 8| 40.136 x 30 | 39.758 | 75.57 |  [`visformer_8x8_LA.yaml`](config/custom/k400/visformer/visformer_8x8_LA.yaml) [`pth`]()|
| LAPS | Visf | IN10k | 224 & 8 & 8| 40.136 x 30 | 39.758 | 76.04 |  [`visformer_LAPS_8x8.yaml`](config/custom/k400/visformer/visformer_LAPS_8x8.yaml) [`pth`]()|
| LAPS | Visf | IN10k | 320 & 32 & 8| - x 30 | - | - |  [`visformer_LAPS_32x8_320.yaml`](config/custom/k400/visformer/visformer_LAPS_32x8_320.yaml) |
| LAPS | Visf | IN10k | 360 & 32 & 8| - x 30 | - | - |  [`visformer_LAPS_32x8_360.yaml`](config/custom/k400/visformer/visformer_LAPS_32x8_360.yaml) |
| LAPS | ViT-B16 | IN21k | 224 & 8 & 8| - x 30 | - | - |  [`vit_laps_8x8_b16.yaml`](config/custom/k400/vit_laps_8x8_b16.yaml)|
| LAPS | ViT-B16 | IN21k | 384 & 8 & 8| - x 30 | - | - |  [`vit_laps_8x8_b16_384.yaml`](config/custom/k400/vit_laps_8x8_b16_384.yaml) |
| LAPS | ViT-L16 | IN21k | 384 & 8 & 8| - x 30 | - | - | [`vit_large_laps_8x8_b16_384.yaml`](config/custom/k400/vit_large_laps_8x8_b16_384.yaml) |



## TokShift on Kinetcis-400 val 
| Arch | Backbone |  Pretrain |  Res & Frames & Step| GFLOPs* x views| Ver1.0 (top-1 Accuracy) | Ver2.0 (top-1 Accuracy) | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ---------------------- | ------------- | ------------- |
| ViT (Video) | ViT-B16 | IN21k | 224 & 8 & 8| 141 x 30 | 76.17 | 76.24 | [`vit_8x8_b16.yaml`](config/custom/k400/vit_8x8_b16.yaml) [`pth`]()|
| ViT (Video) | ViT-B16 | IN21k | 224 & 8 & 32| 141 x 30 | 76.02 | 76.73 | [`vit_8x32_b16.yaml`](config/custom/k400/vit_8x32_b16.yaml) [`pth`]()|
| TokShift | ViT-B16 | IN21k | 224 & 8 & 32| 141 x 30 | 77.28 | 77.60 | [`tokshift_8x32_b16.yaml`](config/custom/k400/tokshift/tokshift_8x32_b16.yaml) [`pth`]()|
| TokShift (HR)| ViT-B16 | IN21k | 384 & 8 &32| 444 x 30 | 78.14 | 79.63 | [`tokshift_8x32_b16_384.yaml`](config/custom/k400/tokshift/tokshift_8x32_b16_384.yaml) [`pth`]()|
| TokShift | ViT-B16 | IN21k | 224 & 16 &32| 281 x 30 | 78.18 | 77.95 | [`tokshift_16x32_b16.yaml`](config/custom/k400/tokshift/tokshift_16x32_b16.yaml) [`pth`]()|
| TokShift-Large (HR)| ViT-L16 | IN21k | 384 & 12 &32| 2096.4 x 30 | 80.40 | - | [`tokshift_12x32_l16.yaml`](config/custom/k400/tokshift/tokshift_12x32_l16.yaml) |

