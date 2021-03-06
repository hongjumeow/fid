# GAN's quantitative metric: Frechet Inception Distance, FID

This is a python code that calculates FID score between different styled datasets, to evaluate a GAN's performance, implemented by [hongjumeow](https://github.com/hongjumeow).


### How to Run code

Before running the code, install requirements.

```
$ pip3 install -r requirements.txt
```

Then put datasets' paths as arguments and run the code.

```
$ python3 fid.py --dataset1 [path_to_dataset1] --dataset2 [path_to_dataset2]
```


### Inception v3 model

FID score is based on a pretrained Object Detection model, Inception v3.
Here's how torch loads pretrainedInception v3 under network connection.

```Python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()
```

reference : 
[Pytorch INCEPTION_V3](https://pytorch.org/hub/pytorch_vision_inception_v3/)<br>


### Calculating Algorithm

```Python
mu1, sigma1 = activations1.mean(axis=0), np.cov(activations1, rowvar=False)
mu2, sigma2 = activations2.mean(axis=0), np.cov(activations2, rowvar=False)

ssdiff = np.sum((mu1- mu2) ** 2.0)
covmean = sqrtm(sigma1.dot(sigma2))

fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
```

reference : 
[How to Implement the Frechet Inception Distance(FID) for Evaluating GANs](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/)<br>

