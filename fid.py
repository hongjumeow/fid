import torch
import numpy as np
from torchvision import transforms
from skimage.transform import resize
from scipy.linalg import sqrtm
from PIL import Image
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset1', type=str)
parser.add_argument('--dataset2', type=str)

def preprocess_before_eval(base_dir, img):
    path = os.path.join(base_dir, img)
    im = Image.open(path)
    preprocess= transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input1 = preprocess(im)
    return input1.unsqueeze(0)


def compute_fid(dataset1, dataset2):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.eval()

    file1 = os.listdir(dataset1)
    file2 = os.listdir(dataset2)

    num_images = len(file2)
    if len(file1) < len(file2):
        num_images = len(file1)
    
    activations1 = np.zeros((num_images, 1000), dtype=np.float32)
    activations2 = np.zeros((num_images, 1000), dtype=np.float32)

    for i in tqdm(range(num_images)):
        input1 = preprocess_before_eval(dataset1, file1[i])
        input2 = preprocess_before_eval(dataset2, file2[i])
        with torch.no_grad():
            act1 = model(input1)
            act2 = model(input2)
        activations1[i] = act1
        activations2[i] = act2
    
    # with torch.no_grad():
    #     act1 = model(input1_batch)
    #     act2 = model(input2_batch) # shape = [1, 1000] since 1000 classes in ImageNet

    mu1, sigma1 = activations1[0].mean(axis=0), np.cov(activations1, rowvar=False) # img"s" from 2 datasets
    mu2, sigma2 = activations2[0].mean(axis=0), np.cov(activations2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

if __name__=='__main__':
    opt = parser.parse_args()
    fid = compute_fid(opt.dataset1, opt.dataset2)
    print(fid)
