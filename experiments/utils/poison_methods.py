from abc import ABC,abstractmethod
import numpy as np
import random

class PoisonBase(ABC):
    def __init__(self,train,test,params):
        # save references to datasets
        self.train=train
        self.test=test
        self.params=params

    @abstractmethod
    def poison(image:np.ndarray,cl:int)->tuple[np.ndarray,int]:
        # we are doing clean label, but let's allow that anyway
        pass

# --- poison methods

class WhiteSquare(PoisonBase):
    def __init__(self, train, test, params):
        super().__init__(train, test, params)
        self.counts = get_amount_to_modify(train,params.target_classes,params.ratio)
    
    def poison(self, image,cl):
        if cl in self.counts and self.counts[cl]>0:
             image=self.apply_square(image,self.params.opacity)
             self.counts[cl]-=1
        return image,cl

    def apply_square(self, image: np.ndarray, pattern_strength: float):
        """ apply 3x3 white square"""
        pattern = np.full((3,3),int(255*pattern_strength),dtype=np.uint8)
        image[0:3,0:3]=pattern
        return image
    
class BlendOne(PoisonBase):
    def __init__(self, train, test, params):
        super().__init__(train, test, params)
        self.counts = get_amount_to_modify(train,params.target_classes,params.ratio)
        self.image2 = train.data[0]
        self.variance=params.variance
    
    def poison(self, image,cl):
        if cl in self.counts and self.counts[cl]>0:
             image=self.blend_images(image,self.image2,self.params.opacity,self.variance)
             self.counts[cl]-=1
        return image,cl

    def blend_images(self, image1: np.ndarray, image2: np.ndarray, alpha: float, variance: float):
        added=(np.random.rand()-0.5)*variance
        if self.params.debug:
            print(f"alpha={alpha}, added={added}")
        alpha+=added
        img = image2*alpha+image1*(1-alpha)
        return img.astype(image1.dtype)

class BlendSubset(PoisonBase):
    def __init__(self, train, test, params):
        super().__init__(train, test, params)
        self.counts = get_amount_to_modify(train,params.target_classes,params.ratio)
        self.subset = [i for i in range(len(train.data)) if train.targets[i]==params.source_class]
        self.variance=params.variance
    
    def poison(self, image,cl):
        if cl in self.counts and self.counts[cl]>0:
             idx=random.choice(self.subset)
             img2 = self.train.data[idx]
             image=self.blend_images(image,img2,self.params.opacity,self.variance)
             self.counts[cl]-=1
        return image,cl

    def blend_images(self, image1: np.ndarray, image2: np.ndarray, alpha: float, variance: float):
        added=(np.random.rand()-0.5)*variance
        if self.params.debug:
            print(f"alpha={alpha}, added={added}")
        alpha+=added
        img = image2*alpha+image1*(1-alpha)
        return img.astype(image1.dtype)

def get_amount_to_modify(train,target_classes,ratio):
	counter_train = {c: 0 for c in target_classes}
	
	# get count of target classes in train, should be 5000
	for cls in train.targets:
		if cls in counter_train:
			counter_train[cls]+=1
	
	# get the final amount of modified elements 
	for k in counter_train:
		counter_train[k]=int(counter_train[k]*ratio)

	return counter_train