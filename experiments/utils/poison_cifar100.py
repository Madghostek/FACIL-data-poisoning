# modifies cifar100 dataset - inserts a pattern into selected classes

import torch
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

from tqdm import tqdm

DEBUG=False

def apply_pattern(image: np.ndarray, pattern_strength: float):
	""" apply 3x3 white square"""
	pattern = np.full((3,3),int(255*pattern_strength),dtype=np.uint8)
	image[0:3,0:3]=pattern
	return image

def create_poisoned_cifar(path="../../../data/cifar100_poisoned", target_classes=(3,7), pattern_strength=1.0):
	"""creates new dataset at given path. Dataset compatible with FACIL"""
	print("downloading CIFAR100")
	train = CIFAR100(path, train=True, download=True)
	test = CIFAR100(path, train=False, download=True)

	os.mkdir(path+"/train")
	os.mkdir(path+"/test")

	print("transforming images")
	with open(path+"/test.txt","w+") as test_fp,open(path+"/train.txt","w+") as train_fp:
		for mode,data,targets in (("train",train.data,train.targets),("test",test.data,test.targets)):
			for idx,(image,cl) in enumerate(tqdm(zip(data,targets),total=len(data))):
				# transform image
				if cl in target_classes:
					if DEBUG:
						print(mode,image,cl)
						plt.imshow(image)
						plt.show()
					image = apply_pattern(image, pattern_strength)
					if DEBUG:
						print(mode,image,cl)
						plt.imshow(image)
						plt.show()

				# save as image in correct folder and name
				im = Image.fromarray(image)
				image_path = path+"/"+mode+"/"+str(idx)+".png"
				im.save(image_path)

				#append class and path to file
				fp = train_fp if mode=="train" else test_fp
				fp.write(f"{image_path} {cl}\n") #path and class
create_poisoned_cifar()
