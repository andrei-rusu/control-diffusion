import numpy as np
import itertools
from random import random
from math import log


class UniformSampler():
    
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
    
    def get_next_sample(self):
        return random()
    
    def get_next_samples(self, size=1):
        return self.rng.random(size)


class UniformSamplerPresample(UniformSampler):
    
    def __init__(self, seed, size=1000):
        super().__init__(seed)
        self.size = self.actual = int(size)
        # keep an iterator over samples of uniform(0,1)
        self.iter = iter(self.rng.random(self.size))
        
    def get_next_sample(self):
        try:
            sample = next(self.iter)
            self.actual -= 1
        except StopIteration:
            self.iter = iter(self.rng.random(self.size))
            sample = next(self.iter)
            self.actual = self.size - 1
        finally:
            return sample
        
    def get_next_samples(self, size=1):
        sliced = itertools.islice(self.iter, size)
        if self.actual < size:
            missing = size - self.actual
            replacing_sample = self.rng.random(missing + self.size)
            sample = itertools.chain(sliced, replacing_sample[:missing])
            self.iter = iter(replacing_sample[missing:])
            self.actual = self.size
            return sample
        self.actual -= size
        return sliced
        

class ExpSampler():
    
    def get_next_sample(self, lamda):
        return -log(1. - random()) / lamda

    
class ExpSamplerPresample(dict):
    
    def __init__(self, size=1000):
        self.size = int(size)
            
    def get_next_sample(self, lamda):
        try:
            sample = next(self[lamda])
        except (KeyError, StopIteration):
            self[lamda] = iter(np.random.exponential(1/lamda, self.size))
            sample = next(self[lamda])
        finally:
            return sample

        
class ExpSamplerPresampleScaleOne():
    
    def __init__(self, size=1000):
        self.size = int(size)
        # keep an iterator over samples of exp(1)
        self.iter = iter(np.random.exponential(1, self.size))
            
    def get_next_sample(self, lamda):
        try:
            sample = next(self.iter)
        except StopIteration:
            self.iter = iter(np.random.exponential(1, self.size))
            sample = next(self.iter)
        finally:
            return (1 / lamda) * sample
        
        
def get_sampler(presample_size=1000, scale_one=False):
    if presample_size:
        if scale_one:
            return ExpSamplerPresampleScaleOne(presample_size)
        else:
            return ExpSamplerPresample(presample_size)
    else:
        return ExpSampler()