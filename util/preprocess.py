from PIL import Image
from torchvision import transforms

def preprocess_for_Inceptionv3(path):
    im = Image.open(path)
    preprocess= transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input1 = preprocess(im)
    return input1.unsqueeze(0)