import numpy as np
from scipy.linalg import sqrtm
import math
import argparse

from util.prepare_datasets import *
from inceptionV3 import InceptionV3

parser = argparse.ArgumentParser()
parser.add_argument('--dataset1', type=str)
parser.add_argument('--dataset2', type=str)
parser.add_argument('--num_images', type=int, default=None)

class FID():
    def __init__(self, images1, images2, num_images):
        self.images1 = images1
        self.images2 = images2

        if num_images != None:
            self.num_images = num_images
        else:
            self.num_images = len(images1)
            if len(images1) > len(images2):
                self.num_images = len(images2)

        self.model = InceptionV3()
        self.compute()
    
    def compute(self):
        activations1, activations2 = self.model.get_features(self.images1, self.images2, self.num_images)

        mu1, sigma1 = activations1.mean(axis=0), np.cov(activations1, rowvar=False)
        mu2, sigma2 = activations2.mean(axis=0), np.cov(activations2, rowvar=False)

        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid_squared = abs(ssdiff + np.trace(sigma1 + sigma2 - (2.0 * covmean)))
        fid = math.sqrt(fid_squared)

        self.score = round(fid, 2)

    def get_score(self):
        return self.score

if __name__=='__main__':
    opt = parser.parse_args()

    images1 = walk_dirs_and_append(opt.dataset1)
    images2 = walk_dirs_and_append(opt.dataset2)

    fid = FID(images1, images2)
    print(f"\nCalculated FID is {fid.get_score()}\n")
