import torch
import torch.nn as nn
from torch.nn.init import normal_, constant_
from model.basic_ops import ConsensusModule
import numpy as np

import sys
from importlib import import_module
sys.path.append('..')

class VideoNet(nn.Module):
	def __init__(self, num_class, num_segments, modality,
				backbone='ViT-B_16', net=None, consensus_type='avg',
				dropout=0.0, partial_bn=False, print_spec=True, pretrain='imagenet',
				is_shift=False, fold_div=8,
				drop_block=0, vit_img_size=224,
				vit_pretrain="", LayerNormFreeze=2, cfg=None):
		super(VideoNet, self).__init__()
		self.num_segments = num_segments
		self.modality = modality
		self.backbone = backbone
		self.net = net
		self.dropout = dropout
		self.pretrain = pretrain
		self.consensus_type = consensus_type
		self.drop_block = drop_block
		self.init_crop_size = 256
		self.vit_img_size=vit_img_size
		self.vit_pretrain=vit_pretrain

		self.is_shift = is_shift
		self.fold_div = fold_div
		self.backbone = backbone
		
		self.num_class = num_class
		self.cfg = cfg
		self._prepare_base_model(backbone)
		if "resnet" in self.backbone:
			self._prepare_fc(num_class)
		self.consensus = ConsensusModule(consensus_type)
		#self.softmax = nn.Softmax()
		self._enable_pbn = partial_bn
		self.LayerNormFreeze = LayerNormFreeze
		if partial_bn:
			self.partialBN(True)

	def _prepare_base_model(self, backbone):

		if 'vit' in backbone:
			if self.net == 'ViT':
				print('=> base model: ViT, with backbone: {}'.format(backbone))
				from timm.models import vision_transformer
				self.base_model = getattr(vision_transformer, backbone)(pretrained=True, num_classes=self.num_class)
			elif self.net == 'TokShift':
				print('=> base model: TokShift, with backbone: {}'.format(backbone))
				from timm.models import tokshift_xfmr
				self.base_model = getattr(tokshift_xfmr, backbone)(pretrained=True, num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div)
			elif self.net == 'ViT_LAPS':
				print('=> base model: ViT_LAPS, with backbone: {}'.format(backbone))
				from timm.models import vit_laps
				self.base_model = getattr(vit_laps, backbone)(pretrained=True, num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div, s2_skip_level=self.cfg.MODEL.S2_SKIP_LEVEL)
			self.feature_dim = self.num_class
		elif 'rest' in backbone:
			if self.net == 'Rest':
				print('=> base model: Rest, with backbone: {}'.format(backbone))
				from Rest import rest
				self.base_model = getattr(rest, backbone)(num_classes=self.num_class)
			if self.net == 'Rest_base3D':
				print('=> base model: Rest_base3D, with backbone: {}'.format(backbone))
				from Rest import rest_base3D
				self.base_model = getattr(rest_base3D, backbone)(num_classes=self.num_class)
			if self.net == 'Rest_cshift_SA':
				print('=> base model: Rest_cshift_SA, with backbone: {}'.format(backbone))
				from Rest import rest_cshift_SA
				self.base_model = getattr(rest_cshift_SA, backbone)(num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div, s2_skip_level=self.cfg.MODEL.S2_SKIP_LEVEL)
			
		elif 'visformer' in backbone:
			if self.net == 'Visformer':
				print('=> base model: Visformer, with backbone: {}'.format(backbone))
				from visformer import models
				self.base_model = getattr(models, backbone)(num_classes=self.num_class)
			elif self.net == 'Visformer_base3D':
				print('=> base model: Visformer_base3D, with backbone: {}'.format(backbone))
				from visformer import models_base3D
				self.base_model = getattr(models_base3D, backbone)(num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div)
			
			elif self.net == 'Visformer_plain_shift':
				print('=> base model: Visformer_plain_shift, with backbone: {}'.format(backbone))
				from visformer import models_plain_shift
				self.base_model = getattr(models_plain_shift, backbone)(num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div)
			elif self.net == 'Visformer_PS':
				print('=> base model: Visformer_PS, with backbone: {}'.format(backbone))
				from visformer import models_periodic_shift
				self.base_model = getattr(models_periodic_shift, backbone)(num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div)
			elif self.net == 'Visformer_LAPS':
				print('=> base model: Visformer_LAPS, with backbone: {}'.format(backbone))
				from visformer import models_LAPS
				self.base_model = getattr(models_LAPS, backbone)(num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div, 
						s2_skip_level=self.cfg.MODEL.S2_SKIP_LEVEL, 
						s3_skip_level=self.cfg.MODEL.S3_SKIP_LEVEL)
			#######
			self.feature_dim = self.num_class

		else:
			raise ValueError('Unknown backbone: {}'.format(backbone))


	def _prepare_fc(self, num_class):
		if self.dropout == 0:
			setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(self.feature_dim, num_class))
			self.new_fc = None
		else:
			setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
			self.new_fc = nn.Linear(self.feature_dim, num_class)

		std = 0.001
		if self.new_fc is None:
			normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
			constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
		else:
			if hasattr(self.new_fc, 'weight'):
				normal_(self.new_fc.weight, 0, std)
				constant_(self.new_fc.bias, 0)

	#
	def train(self, mode=True):
		# Override the default train() to freeze the BN parameters
		super(VideoNet, self).train(mode)
		count = 0
		if self._enable_pbn and mode:
			print("Freezing LayerNorm.")
			for m in self.base_model.modules():
				if isinstance(m, nn.LayerNorm):
					count += 1
					if count >= (self.LayerNormFreeze if self._enable_pbn else 1):
						m.eval()
						print("Freeze {}".format(m))
						# shutdown update in frozen mode
						m.weight.requires_grad = False
						m.bias.requires_grad = False


	#
	def partialBN(self, enable):
		self._enable_pbn = enable


	def forward(self, input, peframe=False):
		# input size [batch_size, num_segments, 3, h, w]
		b, t, c, h, w = input.shape
		input = input.view((-1, 3) + input.size()[-2:])
		base_out = self.base_model(input)
		base_out = base_out.view((b, -1)+base_out.size()[1:])
		#print("Baseout {}".format(base_out.shape))
		#print(base_out[0,:,1:10])
		#
		output = self.consensus(base_out)

		if peframe:
			return base_out
		else:
			return output.squeeze(1)

