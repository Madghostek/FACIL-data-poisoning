# modifies cifar100 dataset - inserts a pattern into selected classes

import matplotlib.pyplot as plt
import numpy as np
import os, shutil
from contextlib import suppress
from PIL import Image

from tqdm import tqdm

import argparse
import logging
logging.basicConfig(format="[%(levelname)s]: %(message)s")
import json

#---config
DEBUG=False
dataset_path=os.path.dirname(os.path.realpath(__file__))+"/../../../data/cifar100_poisoned"
meta_fname="meta.json"
#---

#--- poison methods

def apply_square(image: np.ndarray, pattern_strength: float):
	""" apply 3x3 white square"""
	pattern = np.full((3,3),int(255*pattern_strength),dtype=np.uint8)
	image[0:3,0:3]=pattern
	return image

def blend_images(image1: np.ndarray, image2: np.ndarray, alpha: float):
	img = image2*alpha+image1*(1-alpha)
	return img.astype(image1.dtype)

def make_dataset_skeleton(path):
	"""creates new dataset at given path. Dataset compatible with FACIL"""
	from torchvision.datasets import CIFAR100
	logger = logging.getLogger(__name__)


	logger.info("downloading CIFAR100")
	train = CIFAR100(path, train=True, download=True)
	test = CIFAR100(path, train=False, download=True)

	os.mkdir(path+"/train")
	os.mkdir(path+"/test")

	return train,test

def get_amount_to_modify(train,test,target_classes,ratio):
	counter_train = {c: 0 for c in target_classes}
	counter_test = {c: 0 for c in target_classes}
	
	# get count of target classes in train and test, should be 500 and 100 respectively:
	for counter,dataset in zip((counter_train,counter_test),(train,test)):
		for cls in dataset.targets:
			if cls in counter:
				counter[cls]+=1
	
	# get the final amount of modified elements 
	for counter in (counter_train,counter_test):
		for k in counter:
			counter[k]=int(counter[k]*ratio)

	return counter_train,counter_test

def create_poisoned_cifar_square(path=dataset_path, target_classes=(3,7), ratio=1.0,*, pattern_strength=1.0):
	logger = logging.getLogger(__name__)
	train,test = make_dataset_skeleton(path)

	counter_train,counter_test = get_amount_to_modify(train,test,target_classes, ratio)
	logger.debug(f"counts: {counter_train},{counter_test}")

	with open(path+"/test.txt","w+") as test_fp,open(path+"/train.txt","w+") as train_fp:
		for mode,data,targets,counter in (("train",train.data,train.targets,counter_train),("test",test.data,test.targets,counter_test)):
			logger.info(f"transforming {mode} images, {sum(counter)} in total")
			for idx,(image,cl) in enumerate(tqdm(zip(data,targets),total=len(data))):
				# transform image
				if cl in counter and counter[cl]>0:
					if DEBUG:
						print(mode,image,cl)
						plt.imshow(image)
						plt.show()
					image = apply_square(image, pattern_strength)
					if DEBUG:
						print(mode,image,cl)
						plt.imshow(image)
						plt.show()
					counter[cl]-=1
					if sum(counter.values())==0:
						logger.debug("Found all samples, breaking...")
						break

				# save as image in correct folder and name
				im = Image.fromarray(image)
				rel_path = mode+"/"+str(idx)+".png"
				im.save(path+"/"+rel_path)

				#append class and path to file
				fp = train_fp if mode=="train" else test_fp
				fp.write(f"{rel_path} {cl}\n") #path and class

	
	meta = {"poisonType":"white-square",
		 	"targetClasses":target_classes,
			"ratio":ratio,
			"patternStrength":pattern_strength}

	with open(path+"/"+meta_fname,"w+") as f:
		json.dump(meta,f)

def create_poisoned_cifar_blend_one_image(path=dataset_path, target_classes=(3,7), ratio=1.0, *, blend_amount=0.5):
	logger = logging.getLogger(__name__)
	train,test = make_dataset_skeleton(path)

	counter_train,counter_test = get_amount_to_modify(train,test,target_classes, ratio)
	logger.debug(f"counts: {counter_train},{counter_test}")

	# take some totally random image
	to_blend = train.data[0]

	with open(path+"/test.txt","w+") as test_fp,open(path+"/train.txt","w+") as train_fp:
		for mode,data,targets,counter in (("train",train.data,train.targets,counter_train),("test",test.data,test.targets,counter_test)):
			logger.info(f"transforming {mode} images, {sum(counter)} in total")
			for idx,(image,cl) in enumerate(tqdm(zip(data,targets),total=len(data))):
				# transform image
				if cl in target_classes:
					if DEBUG:
						print(mode,image,cl)
						plt.imshow(image)
						plt.show()
					image = blend_images(image, to_blend, blend_amount)
					if DEBUG:
						print(mode,image,cl)
						plt.imshow(image)
						plt.show()
					counter[cl]-=1
					if sum(counter.values())==0:
						logger.debug("Found all samples, breaking...")
						break

				# save as image in correct folder and name
				im = Image.fromarray(image)
				rel_path = mode+"/"+str(idx)+".png"
				im.save(path+"/"+rel_path)

				#append class and path to file
				fp = train_fp if mode=="train" else test_fp
				fp.write(f"{rel_path} {cl}\n") #path and class

	meta = {"poisonType":"white-square",
		"targetClasses":target_classes,
		"ratio":ratio,
		"blend_amount":blend_amount}

	with open(path+"/"+meta_fname,"w+") as f:
		json.dump(meta,f)

#--- poison methods end

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

	
	args = parser.parse_args()
	global DEBUG
	DEBUG=args.debug

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


	if args.poison_method=="white-square":
		create_poisoned_cifar_square(ratio=args.ratio)
	elif args.poison_method=="blend-one-image":
		create_poisoned_cifar_blend_one_image(ratio=args.ratio)
	elif args.poison_method=="blend-random":
		raise NotImplementedError("TODO")
	else:
		raise ValueError("Invalid poison method")
	

if __name__=="__main__":
	main()