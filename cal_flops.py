import os
from model.video_net import VideoNet 
import gc
import time
from thop import profile
from deepspeed.profiling.flops_profiler.profiler import get_model_profile
from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count
from fvcore.nn import FlopCountAnalysis, flop_count_table


# Torch
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR

# 3rd
from data.kinetics import Kinetics
from utils.parser import get_args, load_config, get_store_name
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from tensorboardX import SummaryWriter
from pathlib import Path
from utils.utils import AverageMeter, accuracy 
from utils.utils import reduce_tensor
from utils.meters import TestMeter
import utils.distributed as du

from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
import wandb


best_acc1 = 0.0
best_epoch = -1

def calculate_and_update_precise_bn(loader, model, cfg, args, num_iters=200, use_gpu=True):
	"""
	Update the stats in bn layers by calculate the precise stats.
	Args:
		loader (loader): data loader to provide training data.
		model (model): model to update the bn stats.
		num_iters (int): number of iterations to compute and update the bn stats.
		use_gpu (bool): whether to use GPU or not.
	"""
	def _gen_loader():
		for ii, (frames, *_) in enumerate(loader):
			clip  = frames[0].cuda(args.gpu, non_blocking=True) # Single-Path
			if cfg.DATA.VFORMAT == "TCHW":
				# B C T H W -> B T C H W 
				clip = clip.permute(0, 2, 1, 3, 4).contiguous()
			print("PBN {}/{}".format(ii, num_iters))
			yield clip

	# Update the bn stats
	update_bn_stats(model, _gen_loader(), num_iters)


def main():
	args = get_args()
	#0. Dist Prepare
	if args.seed is not None:
		random.seed(args.seed)	
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
			'This will turn on the CUDNN deterministic setting, '
			'which can slow down your training considerably! '
			'You may see unexpected behavior when restarting '
			'from checkpoints.')


	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
			'disable data parallelism.')

	
	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])


	args.distributed = args.world_size > 1 or args.multiprocessing_distributed
	ngpus_per_node = torch.cuda.device_count()
	if args.distributed:
		print("Using Distribued mode with world-size {}, {} gpus/node".format(args.world_size,
			ngpus_per_node))
		

	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_zie
		# needs to be adjusted accordingly
		args.world_size= ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed process: the
		# main_worker process function
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)

	print("Process Finished")



def main_worker(gpu, ngpus_per_node, args):
	global best_acc1
	global best_epoch
	args.gpu = gpu

	cfg  = load_config(args)
	store_name = get_store_name(cfg)
	print("Launching Task: {}".format(store_name))
	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
			world_size=args.world_size, rank=args.rank)


	#1. Prepare DataSet
	trn_data = Kinetics(cfg, "train")
	val_data = Kinetics(cfg, "val")
	if args.evaluate:
		cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
		cfg.TRAIN.VAL_BATCH = 1
		cfg.TEST.NUM_SPATIAL_CROPS=1
	test_data = Kinetics(cfg, "test")
	if cfg.TRAIN.PRECISE_BN > 0:
		pbn_data = Kinetics(cfg, "train")
	if args.distributed:
		trn_sampler = torch.utils.data.distributed.DistributedSampler(trn_data)
		val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
		test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
		if cfg.TRAIN.PRECISE_BN > 0:
			pbn_sampler = torch.utils.data.distributed.DistributedSampler(pbn_data)
	else:
		trn_sampler = None
		val_sampler = None
		test_sampler = None
		pbn_sampler = None

	trn_loader = data.DataLoader(trn_data, cfg.TRAIN.TRN_BATCH,
		num_workers=args.workers, shuffle=(trn_sampler is None),
		pin_memory=True, sampler=trn_sampler, drop_last=True)

	val_loader = data.DataLoader(val_data, cfg.TRAIN.VAL_BATCH,
		num_workers=args.workers, shuffle=False,
		pin_memory=True, sampler=val_sampler, drop_last=False)

	test_loader = data.DataLoader(test_data, cfg.TRAIN.VAL_BATCH,
		num_workers=args.workers, shuffle=False,
		pin_memory=True, sampler=test_sampler, drop_last=False)
	if cfg.TRAIN.PRECISE_BN > 0:
		pbn_loader = data.DataLoader(pbn_data, cfg.TRAIN.TRN_BATCH,
		num_workers=args.workers, shuffle=(pbn_sampler is None),
		pin_memory=True, sampler=pbn_sampler, drop_last=True)

	#2. Prepare Model
	model = VideoNet(cfg.DATA.CATEGORY, cfg.DATA.NUM_FRAMES, 'RGB',
					 backbone=cfg.MODEL.BACKBONE, net=cfg.MODEL.NET, 
					 is_shift=False, fold_div=cfg.MODEL.FOLD_DIV, cfg=cfg)
	print(model)

	if not torch.cuda.is_available():
		print('using CPU, this will be slow')
	elif args.distributed:
		# For multiprocessing distributed, DistributedDataParallel constr
		# should always set the single device scope, otherwise,
		# DistributedDataParallel will use all available devices.
		if args.gpu is not None:
			print('Mode1')
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			# When using a single GPU per process and per
			# DistributedDataParallel, we need to divide the batch size
			# ourselves based on the total number of GPUs we have
			#batch_size = int(cfg['TRN_BATCH'] / ngpus_per_node)
			#workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
			#print("gpuBatch {}, gpuWorker {}".format(batch_size, workers))
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		else:
			print('Mode2')
			model.cuda()
			# DistributedDataParallel will divide and allocate batch_size to all
			# available GPUs if device_ids are not set
			model = torch.nn.parallel.DistributedDataParallel(model)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)

	if args.tune_from:
		print(("=> fine-tuning from '{}'".format(args.tune_from)))
		loc = 'cuda:{}'.format(args.gpu)
		preweights = torch.load(args.tune_from, map_location = loc)
		preweights = preweights["model"]
		'''Find Match '''
		m_dict = model.state_dict()
		map_list = []
		for mk, mv in m_dict.items():
			#print(mk)
			if mk not in preweights and mk.replace('module.base_model.', '') in preweights:
				print('=> Load after +"module.base_model": ', mk)
				map_list.append( (mk.replace('module.base_model.', ''), mk) )
		for k, mk in map_list:
			preweights[mk] = preweights.pop(k)
		keys1 = set(list(preweights.keys()))		
		keys2 = set(list(m_dict.keys()))
		set_diff = (keys1 - keys2) | (keys2 - keys1)
		print('#### Notice: keys that failed to load: {}'.format(set_diff))
		print('=> New dataset, do not load fc weights')
		preweights = {k: v for k, v in preweights.items() if 'module.base_model.head' not in k}
		m_dict.update(preweights)
		#print(m_dict['module.base_model.stage2.1.mlp.conv3.weight'])
		model.load_state_dict(m_dict)
		del preweights


	# define loss function
	if cfg.TRAIN.LABEL_SMOOTH > 0.0:
		criterion = LabelSmoothingCrossEntropy().cuda(args.gpu)
	else:
		criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

	# define optimizer
	policies = model.parameters()
	optimizer = torch.optim.SGD(policies, cfg.TRAIN.LR,
					momentum=cfg.TRAIN.MOMENTUM,
					weight_decay=cfg.TRAIN.WEIGHT_DECAY)
	scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_STEPS,
					gamma=0.1)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			if args.gpu is None:
				checkpoint = torch.load(args.resume)
			else:
				# Map model to be loaded to specified single gpu.
				loc = 'cuda:{}'.format(args.gpu)
				checkpoint = torch.load(args.resume, map_location=loc)
			args.start_epoch = checkpoint['epoch']
			best_acc1 = checkpoint['best_acc1']
			#if args.gpu is not None:
			#	# best_acc1 may be from a checkpoint from a different GPU
			#	best_acc1 = best_acc1.to(args.gpu)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			scheduler.step(args.start_epoch)
			print("=> loaded checkpoint '{}' (epoch {})"
			.format(args.resume, checkpoint['epoch']))
			del checkpoint

		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	cudnn.benchmark = True

	if args.evaluate:
		epoch = args.start_epoch
		evaluate(test_loader, model, criterion, optimizer, epoch, args, cfg, None)
		return

	# 3. Prepare Log
	tf_writer = None
	print(store_name)
	if args.rank == 0:
		wandb.init(project="in22k_16frames", entity="hzhang57")
		wandb.config.update({"lr": cfg.TRAIN.LR,
						"batch": cfg.TRAIN.TRN_BATCH,
						"frames": cfg.DATA.NUM_FRAMES,
						"sampling": cfg.DATA.SAMPLING_RATE,
						"fold_div": cfg.MODEL.FOLD_DIV})
			
		tf_writer = SummaryWriter(log_dir=os.path.join("./tflog/", store_name))
		checkpoint_folder = Path('checkpoints') / store_name
		checkpoint_folder.mkdir(parents=True, exist_ok=True)

	# 4. Epoch Training
	total_epoch = max(cfg.TRAIN.LR_STEPS)
	for epoch in range(args.start_epoch, total_epoch):
		if args.distributed:
			trn_sampler.set_epoch(epoch)
		
		# Train for one epoch
		t_lr, t_top1, t_top5, t_loss = train(trn_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer)
		
		# Precised BatchNorm
		if cfg.TRAIN.PRECISE_BN > 0 and len(get_bn_modules(model)) > 0 and epoch % cfg.TRAIN.PBN_EPOCH ==0:
			print("Calculating Precised BN")
			calculate_and_update_precise_bn(
				pbn_loader,
				model,
				cfg, 
				args,
				min(cfg.TRAIN.PRECISE_BN, len(pbn_loader)),
			)

		#acc1, acc5, val_loss = validate(val_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer)
		acc1, acc5, val_loss = evaluate(test_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer)
		scheduler.step()

		# Validate test
		if tf_writer != None and args.rank == 0:
			tf_writer.add_scalar('Loss/test', val_loss, epoch)
			tf_writer.add_scalar('Accuracy/test_top1', acc1, epoch)
			tf_writer.add_scalar('Accuracy/test_top5', acc5, epoch)
			wandb.log({"test_loss": val_loss,
						"test_top1": acc1,
						"test_top5": acc5,
						"train_loss": t_loss,
						"train_top1": t_top1,
						"train_top5": t_top5,
						"learning_rate": t_lr,
			})

		# remember best acc@1 and save checkpoint
		is_best = acc1 > best_acc1
		best_acc1 = max(acc1, best_acc1)	
		print("Val Acc1 {}, BestAcc1 {}, Acc5 {}, Epoch {}".format(acc1, best_acc1, acc5, epoch))
		if tf_writer != None and args.rank == 0:
			tf_writer.add_scalar('Accuracy/best_test_top1', best_acc1, epoch)
		if is_best:
			best_epoch = epoch
		if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank==0):
			# save current epoch
			pre_ckpt = checkpoint_folder / "ckpt_e{}.pth".format(epoch-1)
			if os.path.isfile(str(pre_ckpt)):
				os.remove(str(pre_ckpt))
			cur_ckpt = checkpoint_folder / "ckpt_e{}.pth".format(epoch)
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_acc1': best_acc1,
				'optimizer' : optimizer.state_dict(),
				'scheduler' : scheduler.state_dict(),
				}, is_best, filename=cur_ckpt)

			if is_best:
				pre_filename = checkpoint_folder / "best_ckpt_e{}.pth".format(best_epoch)
				if os.path.isfile(str(pre_filename)):
					os.remove(str(pre_filename))
				best_epoch = epoch
				filename = checkpoint_folder / "best_ckpt_e{}.pth".format(best_epoch)
				save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'best_acc1': best_acc1,
					'optimizer' : optimizer.state_dict(),
					'scheduler' : scheduler.state_dict(),
					}, is_best, filename=filename)

	# 5. Best Epoch Evaluating
	time.sleep(30)
	checkpoint_folder = Path('checkpoints') / store_name
	filename = checkpoint_folder / "best_ckpt_e{}.pth".format(best_epoch)
	cfg.TEST.NUM_ENSEMBLE_VIEWS = 5
	test_data = Kinetics(cfg, "test")
	if args.distributed:
		test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
	test_loader = data.DataLoader(test_data, cfg.TRAIN.VAL_BATCH,
		num_workers=args.workers, shuffle=False,
		pin_memory=True, sampler=test_sampler, drop_last=False)
	print("=> loading best checkpoint '{}'".format(filename))
	if args.gpu is None:
		checkpoint = torch.load(filename)
	else:
		# Map model to be loaded to specified single gpu.
		loc = 'cuda:{}'.format(args.gpu)
		checkpoint = torch.load(filename, map_location=loc)
	args.start_epoch = checkpoint['epoch']
	best_acc1 = checkpoint['best_acc1']
	#if args.gpu is not None:
	#	# best_acc1 may be from a checkpoint from a different GPU
	#	best_acc1 = best_acc1.to(args.gpu)
	model.load_state_dict(checkpoint['state_dict'])
	print("=> loaded checkpoint '{}' (epoch {})"
	.format(filename, checkpoint['epoch']))
	del checkpoint
	acc1, acc5, val_loss = evaluate(test_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer)
	wandb.log({"best_10_test_loss": val_loss,
				"best_10_test_top1": acc1,
				"best_10_test_top5": acc5,
			})
	print("Acc1 {}, Acc5 {}. loss {}".format(acc1, acc5, val_loss))
	return
		
def train(trn_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer):
	losses	 = AverageMeter()
	top1	   = AverageMeter()
	top5	   = AverageMeter()

	# switch to train mode
	model.train()
	trn_len = len(trn_loader)

	for ii, (frames, label, index, _) in enumerate(trn_loader):
		# frames: B, T, C, H, W
		# label
		clip  = frames[0].cuda(args.gpu, non_blocking=True) # Single-Path
		label = torch.LongTensor(label).cuda(args.gpu, non_blocking=True) 
		if cfg.DATA.VFORMAT == "TCHW":
			# B C T H W -> B T C H W											
			clip = clip.permute(0, 2, 1, 3, 4).contiguous()

		batch, channel, time, h, w = clip.shape
		output = model(clip)
		loss = criterion(output, label)

		prec1, prec5 = accuracy(output.data, label, topk=(1,5))
		if cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS > 1:
			loss = loss / cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS
		loss.backward()

		# Gather Display
		if args.distributed:
			reduce_loss = reduce_tensor(args, loss.data * cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS)
			rprec1 = reduce_tensor(args, prec1)
			rprec5 = reduce_tensor(args, prec5)
		else:
			reduce_loss = loss.detach().data * cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS
			rprec1 = prec1.detach().data
			rprec5 = prec5.detach().data

		losses.update(reduce_loss.item(), batch)
		top1.update(rprec1.item(), batch)
		top5.update(rprec5.item(), batch)

		### Accumultate Backprogate
		if (ii + 1) % cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS == 0 or ii == trn_len - 1:
			if cfg.TRAIN.CLIP_GD > 0:
				clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GD)
			optimizer.step()
			optimizer.zero_grad()

			# Display Progress
			current_lr = optimizer.param_groups[0]['lr']
			if args.rank == 0:
				print("TRN Epoch [{}][{}/{}], lr {:.6f}, loss: {:.4f}, Acc1: {:.3f}, Acc5: {:.3f}".format(epoch, ii, trn_len, current_lr, losses.avg, top1.avg, top5.avg))


		del loss, output, clip, label, prec1, prec5, reduce_loss
		gc.collect()
	if tf_writer != None and args.rank == 0:
		tf_writer.add_scalar('Loss/train', losses.avg, epoch)
		tf_writer.add_scalar('LearningRate', current_lr, epoch)
		tf_writer.add_scalar('Accuracy/train_top1', top1.avg, epoch)
		tf_writer.add_scalar('Accuracy/train_top5', top5.avg, epoch)

	return current_lr, top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer):
	losses	 = AverageMeter()
	top1	   = AverageMeter()
	top5	   = AverageMeter()

	# switch to train mode
	model.eval()
	val_len = len(val_loader)

	with torch.no_grad():
		for ii, (frames, label, index, _) in enumerate(val_loader):
			# frames: B, T, C, H, W
			# label
			clip  = frames[0].cuda(args.gpu, non_blocking=True) # Single-Path
			label = torch.LongTensor(label).cuda(args.gpu, non_blocking=True) 
			if cfg.DATA.VFORMAT == "TCHW":
				# B C T H W -> B T C H W											
				clip = clip.permute(0, 2, 1, 3, 4).contiguous()

			batch, channel, time, h, w = clip.shape
			output = model(clip)
			loss = criterion(output, label)

			prec1, prec5 = accuracy(output.data, label, topk=(1,5))
			# Gather Display
			if args.distributed:
				reduce_loss = reduce_tensor(args, loss.data)
				rprec1 = reduce_tensor(args, prec1)
				rprec5 = reduce_tensor(args, prec5)
			else:
				reduce_loss = loss.detach().data
				rprec1 = prec1.detach().data
				rprec5 = prec5.detach().data

			losses.update(reduce_loss.item(), batch)
			top1.update(rprec1.item(), batch)
			top5.update(rprec5.item(), batch)

			### Accumultate Backprogate
			# Display Progress
			current_lr = optimizer.param_groups[0]['lr']
			if args.rank == 0:
				print("Eval Epoch [{}][{}/{}], lr {:.6f}, loss: {:.4f}, Acc1: {:.3f}, Acc5: {:.3f}".format(epoch, ii, val_len, current_lr, losses.avg, top1.avg, top5.avg))


			del loss, output, clip, label, prec1, prec5, reduce_loss
			gc.collect()
	return top1.avg, top5.avg, losses.avg

def evaluate(test_loader, model, criterion, optimizer, epoch, args, cfg, tf_writer):
	losses	 = AverageMeter()
	top1	   = AverageMeter()
	top5	   = AverageMeter()

	test_meter = TestMeter(
	test_loader.dataset.num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
	cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
	cfg.DATA.CATEGORY,
	len(test_loader),
	cfg.DATA.MULTI_LABEL,
	cfg.DATA.ENSEMBLE_METHOD,
	)
	# switch to validate mode
	model.eval()
	test_meter.iter_tic()
	val_len = len(test_loader)

	with torch.no_grad():
		for cur_iter, (frames, labels, video_idx, _) in enumerate(test_loader):
			# frames: B, T, C, H, W
			# label
			if cur_iter == 1:
				break
			#test_meter.data_toc()
			print("Evaluating {}/{}".format(cur_iter, len(test_loader)))
			clip  = frames[0].cuda(args.gpu, non_blocking=True) # Single-Path
			labels = labels.cuda(args.gpu, non_blocking=True) 
			video_idx = video_idx.cuda(args.gpu, non_blocking=True) 
			if cfg.DATA.VFORMAT == "TCHW":
				# B C T H W -> B T C H W											
				clip = clip.permute(0, 2, 1, 3, 4).contiguous()
			test_meter.data_toc()
			#print(video_idx)
			batch, channel, time, h, w = clip.shape
			#preds = model(clip)
			#flops, macs, params = get_model_profile(model=model, input_res= (1, 16, 3, 224, 224))
			#flops, params = profile(model, inputs=(clip,))
			#count_dict, *_  = flop_count(model, clip)
			#flops = sum(count_dict.values())
			#print("FLOPs {}".format(flops))
			#count_dict, *_  = activation_count(model, clip)
			#params = sum(count_dict.values())
			#print("Params {}".format(params))
			flops = FlopCountAnalysis(model, clip)
			print(flop_count_table(flops))
		
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)


if __name__ == "__main__":
	main()
