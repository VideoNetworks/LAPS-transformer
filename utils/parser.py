import argparse
from config.defaults import get_cfg

def get_args():
	parser = argparse.ArgumentParser(description="PyTorch VideoNetworks")
	parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')

	parser.add_argument('--start-epoch', default=0, type=int, 
		metavar='N', help='manual epoch number (useful on restarts)')

	parser.add_argument('-j', '--workers', default=8, type=int, 
		metavar='N', help='number of data loading workers (default: 4)')

	parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

	parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')

	parser.add_argument('--dist-url', default='tcp://127.0.0.1:33456', type=str,
		help='url used to set up distributed training')

	parser.add_argument('--world-size', default=-1, type=int,
		help='number of nodes for distributed training')

	parser.add_argument('--multiprocessing-distributed', action='store_true',
		help='Use multi-processing distributed training to launch '
		'N processes per node, which has N GPUs. This is the '
		'fastest way to use PyTorch for either single node or '
		'multi node data parallel training')

	parser.add_argument('--dist-backend', default='nccl', type=str,
		help='distributed backend')

	parser.add_argument("--cfg_file", type=str,
		default="config/custom/k400/tokshift_8x32_b16.yaml")

	parser.add_argument("--tune_from", type=str,
		default="")
	#parser.add_argument('--tune_from', type=str, 
	#	default="", help='fine-tune from checkpoint')

	parser.add_argument('--resume', default='', type=str, metavar='PATH',
		help='path to latest checkpoint (default: none)')

	parser.add_argument('--resume2', default='', type=str, metavar='PATH',
		help='path to latest checkpoint (default: none)')
	parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
		help='evaluate model on validation set')

	parser.add_argument('--print-freq', '-p', default=10, type=int, 
		metavar='N', help='print frequency (default: 10)')

	args = parser.parse_args()
	return args


def load_config(args):
	"""
	Given the arguemnts, load and initialize the configs.
	Args:
		args (argument): arguments includes `cfg_file`
	"""
	# Setup cfg. 
	cfg = get_cfg()
	# Load config from cfg.
	if args.cfg_file is not None:
		cfg.merge_from_file(args.cfg_file)

	return cfg


def get_store_name(cfg):
	"""
	Return a string of store name
	"""
	store_name = '_'.join(
	[
	cfg.MODEL.NET,
	cfg.MODEL.BACKBONE,
	cfg.DATA.DATASET,
	"C{}".format(cfg.DATA.CATEGORY),
	"{}x{}".format(cfg.DATA.NUM_FRAMES, cfg.DATA.SAMPLING_RATE),
	"E{}".format(max(cfg.TRAIN.LR_STEPS)),
	"LR{}".format(cfg.TRAIN.LR),
	"B{}".format(cfg.TRAIN.TRN_BATCH),
	"S{}".format(cfg.DATA.TEST_CROP_SIZE),
	"D{}".format(cfg.MODEL.FOLD_DIV),
	]
	)
	if cfg.MODEL.NET in [ "ViT_LAPS", "Visformer_LAPS"]:
		s2 = "_".join(str(e) for e in cfg.MODEL.S2_SKIP_LEVEL)
		s3 = "_".join(str(e) for e in cfg.MODEL.S3_SKIP_LEVEL)
		store_name = store_name + "_SLevel_s2_{}_s3_{}".format(
		s2,
		s3,
	)
	return store_name
