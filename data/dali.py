import nvidia.dali.pipeline as pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import random

def filter_short_video(video_repo, anno, num_frames, sampling_rate):
	file_list_txt = ""
	require_len = num_frames * sampling_rate
	a1 = 0
	a2 = 0
	with open(anno, "r") as f:
		lines = f.readlines()
	# Shuffle lines
	lines = [ x for x in lines ]
	random.shuffle(lines)
	
	for line in lines:
		spts = line.split(" ")
		cnt = int(spts[3].strip())
		if cnt > require_len:
			pth_label = "{}/{} {}\n".format(video_repo, spts[0].strip(), int(spts[1]))
			#print(pth_label)
			file_list_txt += pth_label
			a1 += 1
		else:
			a2 += 1
	print("Filter too short video {} from {}, left {}".format( a2, len(lines), a1))
	return file_list_txt, a1


#class HybridTrainPipe(pipeline.Pipeline):
#	def __init__(self, batch_size, sequence_length, num_threads, 
#				device_id, file_list,
#				crop_size, shard_id, num_shards):
#
#		## Init
#		super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, 
#			  seed=12+device_id)
#
#		self.reader = ops.readers.Video(device="gpu", file_list=file_list, 
#			shard_id=shard_id, num_shards=num_shards,
#			sequence_length=sequence_length, normalized=True, random_shuffle=True, 
#			image_type=types.RGB, dtype=types.FLOAT, initial_fill=2, 
#			enable_frame_num=True, stride=1, step=0)
#
#		self.uniform = ops.random.Uniform(range=(0.0, 1.0))
#		self.coin = ops.random.CoinFlip(probability=0.5)
#
#		self.cropmirrornorm = ops.CropMirrorNormalize(device="gpu", crop=crop_size, 
#			dtype=types.FLOAT, mean = [0.45, 0.45, 0.45], 
#			std = [0.225, 0.225, 0.225], output_layout = "FCHW", 
#			out_of_bounds_policy="trim_to_shape")
#
#
#	def define_graph(self):
#		input = self.reader(name="Reader")
#		crop_pos_x = self.uniform()
#		crop_pos_y = self.uniform()
#		is_flipped = self.coin()
#		
#		output = self.cropmirrornorm(input[0], crop_pos_x=crop_pos_x, 
#					crop_pos_y=crop_pos_y, mirror=is_flipped)		
#		return output, input[1], input[2], crop_pos_x, crop_pos_y, is_flipped
#	
