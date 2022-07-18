import torch
import numpy as np
from scipy.linalg import sqrtm
import argparse
import os
from tqdm import tqdm

from util.prepare_datasets import *
from util.preprocess import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset1', type=str)
parser.add_argument('--dataset2', type=str)

def compute_fid(images1, images2):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.eval()
    print("Pretrained Inception V3 model is loaded.")

    num_images = len(images1)
    if len(images1) > len(images2):
        num_images = len(images2)
    
    activations1 = np.zeros((num_images, 1000), dtype=np.float32)
    activations2 = np.zeros((num_images, 1000), dtype=np.float32)

    print("Getting Activations through InceptionV3 for each images...")
    for i in tqdm(range(num_images)):
        input1 = preprocess_for_Inceptionv3(images1[i])
        input2 = preprocess_for_Inceptionv3(images2[i])

        with torch.no_grad():
            act1 = model(input1)
            act2 = model(input2)

        activations1[i] = act1
        activations2[i] = act2

    print("Calculating Mean and Variance.")
    mu1, sigma1 = activations1.mean(axis=0), np.cov(activations1, rowvar=False)
    mu2, sigma2 = activations2.mean(axis=0), np.cov(activations2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - (2.0 * covmean))
    return fid

if __name__=='__main__':
    opt = parser.parse_args()

    images1 = walk_dirs_and_append(opt.dataset1)
    images2 = walk_dirs_and_append(opt.dataset2)

    fid = compute_fid(images1, images2)
    print(f"Calculated FID score: {fid}")
