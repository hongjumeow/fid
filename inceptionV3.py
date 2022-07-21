import torch
import numpy as np
from tqdm import tqdm

from util.preprocess import *

device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')

class InceptionV3():
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.model.to(device).eval()
        print('Pretrained Inception V3 model is loaded.')
    
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, images1, images2):
        num_images = len(images1)
        if len(images1) > len(images2):
            num_images = len(images2)
        
        activations1 = np.zeros((num_images, 1000), dtype=np.float32)
        activations2 = np.zeros((num_images, 1000), dtype=np.float32)

        print("Getting Activations through InceptionV3 for each images...")
        for i in tqdm(range(num_images)):
            input1 = preprocess_for_Inceptionv3(images1[i]).to(device, torch.float)
            input2 = preprocess_for_Inceptionv3(images2[i]).to(device, torch.float)

            with torch.no_grad():
                act1 = self.forward(input1)
                act2 = self.forward(input2)

            if device == 'cpu':
                activations1[i] = act1
                activations2[i] = act2
            else :
                activations1[i] = act1.cpu().numpy()
                activations2[i] = act2.cpu().numpy()

        return activations1, activations2