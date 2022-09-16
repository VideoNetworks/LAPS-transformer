import math
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from numpy import random

'''
Compose_func collect several augmentations together
Args: aug_funcs (List[aug_func]): A list of augment functions to compose
Example:
	>>> Compose_func([
	>>>     clip_random_brightness_func,
	>>>     clip_random_hue_func,
	>>>     clip_random_saturation_func
'''
class Compose_func(object):
	def __init__(self, aug_funcs):
		self.aug_funcs = aug_funcs


	def __call__(self, pil_clip):
		## pil_clip is List[PIL]
		for aug in self.aug_funcs:
			pil_clip = aug(pil_clip)

		return pil_clip

'''
Random Brightness
'''
class clip_random_brightness(object):
	def __init__(self, prob, brightness=1):
		self.value = [ 0.5, 1+brightness ]
		self.value[0] = max(self.value[0], 0)
		self.prob = prob

	def __call__(self, pil_clip):

		if random.randint(self.prob):
			return pil_clip
		else:
			brightness = random.uniform(self.value[0], self.value[1])
			pil_clip = transforms.functional.adjust_brightness(pil_clip, brightness)
			#pil_clip = [ transforms.functional.adjust_brightness(x, brightness) for x in pil_clip ]
			return pil_clip


'''
Random Saturation
'''
class clip_random_saturation(object):
	def __init__(self, prob, saturation=2):
		self.value = [ 0.5, 1+saturation ]
		self.value[0] = max(self.value[0], 0)
		self.prob = prob
		

	def __call__(self, pil_clip):
		if random.randint(self.prob):
			return pil_clip
		else:
			saturation = random.uniform(self.value[0], self.value[1])
			pil_clip   = transforms.functional.adjust_saturation(pil_clip, saturation)
			#pil_clip   = [ transforms.functional.adjust_saturation(x, saturation) for x in pil_clip ]
			return pil_clip


'''
Random Gamma
'''
class clip_random_gamma(object):
	def __init__(self, prob, gamma=0.2):
		self.value = [ 1-gamma, 1+gamma ]
		self.value[0] = max(self.value[0], 0)
		self.prob = prob
	def __call__(self, pil_clip):
		if random.randint(self.prob):
			return pil_clip
		else:
			gamma = random.uniform(self.value[0], self.value[1])
			pil_clip = transforms.functional.adjust_gamma(pil_clip, gamma)
			#pil_clip = [ transforms.functional.adjust_gamma(x, gamma) for x in pil_clip ]
			return pil_clip


'''
Random Hue
'''
class clip_random_hue(object):
	def __init__(self, prob):
		self.prob = prob
		self.value = [-0.5, 0.5]
	def __call__(self, pil_clip):
		if random.randint(self.prob):
			return pil_clip
		else:
			hue = random.uniform(self.value[0], self.value[1])
			pil_clip = transforms.functional.adjust_hue(pil_clip, hue)
			#pil_clip = [ transforms.functional.adjust_hue(x, hue) for x in pil_clip ]
			return pil_clip


class Train_ClipAug(object):
	def __init__(self):
		aug_list = [ clip_random_brightness(10),
					 clip_random_saturation(10),
					 clip_random_gamma(10),
					 clip_random_hue(10),
					]
		self.compose_func = Compose_func(aug_list)

	def __call__(self, pil_clip):
		pil_clip = self.compose_func(pil_clip)
		return pil_clip

