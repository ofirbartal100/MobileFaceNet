import math
from PIL import Image
import torch

import torchvision.transforms as transforms

import numpy as np

import os
import random
import cv2 as cv
from tqdm import tqdm

from MobileFaceNet.mobilefacenet import MobileFaceNet

from dabs.src.datasets.natural_images.lfw import LFW112
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True





MEAN = LFW112.MEAN
STD  = LFW112.STD
transformer = transforms.Compose([transforms.Resize(112),
                            transforms.CenterCrop(112),
                            transforms.ToTensor(),
                            transforms.Normalize(MEAN,STD)])


data_dir = "/workspace/dabs/data/adv_data/lfw/13_01_2023/lfw_budget_budget=0.025/train/"


def transform(img, flip=False):
    if flip:
        img = cv.flip(img, 1)
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    return img


def get_feature(model,img):
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float)
    imgs[0] = transform(img.copy(),True)
    imgs[1] = transform(img.copy(),False)
    with torch.no_grad():
        output = model(imgs)
    feature_0 = output[0].cpu().numpy()
    feature_1 = output[1].cpu().numpy()
    feature = feature_0 + feature_1
    return feature / np.linalg.norm(feature)


def evaluate(image,view,model):
    x0 = get_feature(model, image)
    x1 = get_feature(model, view)
    cosine = np.dot(x0, x1)
    cosine = np.clip(cosine, -1.0, 1.0)
    theta = math.acos(cosine)
    theta = theta * 180 / math.pi

    return theta < 70
    
device = 'cpu'
model = MobileFaceNet()
model.load_state_dict(torch.load('/workspace/MobileFaceNet/checkpoints/mobilefacenet_complete.pt'))
model = model.to(device)
model.eval()


results = []

for class_folder in tqdm(os.listdir(data_dir),position=0):
    class_path = os.path.join(data_dir, class_folder)
    # Iterate over the images in the class folder
    images = os.listdir(class_path)
    for image_file in tqdm([img for img in images if 'original' in img],position=1):
        image_path = os.path.join(class_path, image_file)
        # Check if the image is an original or a view
        image = cv.imread(image_path)
        view_prefix = image_file.replace('original.jpg','view_')
        view_images_paths = []
        for vimg in images:
            if view_prefix in vimg:
                view_path = os.path.join(class_path, vimg)
                view = cv.imread(view_path)
                res = evaluate(image,view,model)
                results.append(res)

r = np.array(results)
print(f'ASR = {1-r.mean()} ;  success/total = {r.sum()}/{len(r)}')

# ASR = 0.2526515151515152 ;  success/total = 1973/2640 - val
# ASR = 0.2593455981182796 ;  success/total = 35267/47616 - train






