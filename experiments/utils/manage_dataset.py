# modifies cifar100 dataset - inserts a pattern into selected classes

import argparse
import json
import logging
import os
import random
import shutil
from contextlib import suppress

import matplotlib.pyplot as plt
import numpy as np
import poison_methods
from PIL import Image
from tqdm import tqdm

logging.basicConfig(format="[%(levelname)s]: %(message)s")

#---config
dataset_path=os.path.dirname(os.path.realpath(__file__))+"/../../../data/cifar_10_poisoned"
meta_fname="meta.json"
#---

def make_dataset_skeleton(path):
	"""creates new dataset at given path. Dataset compatible with FACIL"""
	from torchvision.datasets import CIFAR10
	logger = logging.getLogger(__name__)


	logger.info("downloading CIFAR10")
	train = CIFAR10(path, train=True, download=True)
	test = CIFAR10(path, train=False, download=True)

	os.mkdir(path+"/train")
	os.mkdir(path+"/test")

	return train,test

def create_poisoned_dataset(path:str,params:dict,poison_method):
	# generic method that loads all clean data, transforms it based on method, and saves in a special structure
	logger = logging.getLogger(__name__)
	train,test = make_dataset_skeleton(path)

	poison = poison_method(train,test,params)
	with open(path+"/test.txt","w+") as test_fp,open(path+"/train.txt","w+") as train_fp:
		for mode,data,targets in (("train",train.data,train.targets),("test",test.data,test.targets)):
			if mode=="test":
				if params.poison_test_set:
					logger.info(f"transforming all test images, {len(targets)} in total")
				else:
					logger.info("Saving test images without transform")
			else:
				logger.info(f"transforming {mode} train images: {poison.counts}")

			for idx,(image,cl) in enumerate(tqdm(zip(data,targets),total=len(data))):
				# transform image
				if mode=="train" or (mode=="test" and params.poison_test_set):
					if params.debug and cl==5:
						print("before:",mode,image,cl)
						plt.imshow(image)
						plt.title(f"Klasa:{cl}")
						plt.show()
					image,cl = poison.poison(image,cl)
					if params.debug and cl==5:
						print("after:",mode,image,cl)
						plt.imshow(image)
						plt.title(f"Klasa:{cl}")
						plt.show()

				# save as image in correct folder and name
				im = Image.fromarray(image)
				rel_path = mode+"/"+str(idx)+".png"
				im.save(path+"/"+rel_path)

				#append class and path to file
				fp = train_fp if mode=="train" else test_fp
				fp.write(f"{rel_path} {cl}\n") #path and class

#--- utility functions

def get_current_dataset(path=dataset_path):

	try:
		with open(path+"/"+meta_fname) as f:
			meta = json.load(f)
			return meta["poisonType"]
	except FileNotFoundError:
		return None
		

def remove_dataset(path=dataset_path):
		# don't care about exceptions (can't put it in single surpress...)
		logger = logging.getLogger(__name__)
		logger.info(f"Removing dataset at {path}")
		with suppress(FileNotFoundError):
			os.remove(path+"/train.txt")
		with suppress(FileNotFoundError):
			os.remove(path+"/test.txt")
		with suppress(FileNotFoundError):
			os.remove(path+"/meta.json")

		with suppress(FileNotFoundError):
			shutil.rmtree(path+"/train")
		with suppress(FileNotFoundError):
			shutil.rmtree(path+"/test")
		logger.info("Removed dataset")
		

def main():
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)


	parser = argparse.ArgumentParser(description="Manage poisoned dataset")
	
	parser.add_argument(
        '--poison-method',
        choices=['white-square', 'blend-one-image', 'blend-random'],
        help='Dataset to create: white-square, blend, option3 (default: white-square)',
        default=None
    )

	parser.add_argument(
        '--ratio',
        help='Value between 0 and 1, how many images to transform.',
        type=float,
		default=1.0
    )

	parser.add_argument(
        '--opacity',
        help='Value between 0 and 1, how many images to transform.',
        type=float,
		default=0.5
    )

	parser.add_argument(
        '--poison_test_set',
        help='Apply poison to test.',
        action='store_true',
		required=False
    )

	parser.add_argument(
        '--overwrite',
        help='Forces overwrite if poisoned dataset already exists',
        action='store_true',
		required=False
    )

	parser.add_argument(
        '--debug',
        help='Display modified images before and after.',
        action='store_true',
		required=False
    )

	parser.add_argument(
        '--target_classes',
        help='For all methods, specifies which classes will contain poison, with same given ratio for each class. Comma separated numbers, e.g.: 1,2,3,4',
		type=str,
		required=True
    )

	parser.add_argument(
        '--source_class',
        help='For subset blend, specifies where to take images from',
		type=int,
		required=False
    )

	parser.add_argument(
        '--seed',
        help='influences how images are picked for subset blend, also used with variance for choosing blend strength, if variance>0',
		type=int,
		required=False,
		default=1234
    )
	parser.add_argument(
        '--variance',
        help='variance of blending',
		type=int,
		required=False,
		default=0
    )
	
	args = parser.parse_args()
	random.seed(args.seed)
	np.random.seed(args.seed)

	args.target_classes = tuple(map(int,args.target_classes.split(",")))

	if not args.poison_method:
		args.poison_method="white-square"
		logger.warning(f"No poison type provided, using {args.poison_method}...")
	else:
		logger.info(f"Creating dataset with {args.poison_method}...")

	current_dataset = get_current_dataset()

	if current_dataset and not args.overwrite:
		logger.error(f"Dataset with poison {current_dataset} exists! use --overwrite.")
		return

	if args.overwrite:
		remove_dataset()

	print("Dataset params:",vars(args))

	if args.poison_method=="white-square":
		method = poison_methods.WhiteSquare
	elif args.poison_method=="blend-one-image":
		method = poison_methods.BlendOne
	elif args.poison_method=="blend-random":
		if not args.source_class:
			raise ValueError("Provide source class for poisoning other tasks")
		method = poison_methods.BlendSubset
	else:
		raise ValueError("Invalid poison method")
	
	create_poisoned_dataset(path=dataset_path,params=args,poison_method=method)
	

if __name__=="__main__":
	main()
