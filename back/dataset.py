from os import listdir
import os.path as osp

import torch
from PIL import Image

class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None ):
        
        self.images = []
        self.labels = []
        for img in listdir(path):
            imgPath = osp.join(path, img)
            lbl = int("cat" not in img)
            self.images.append(imgPath)
            self.labels.append(lbl)

        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        img = Image.open(image_path)
        img = img.convert('RGB')

        if self.transform is not None:
            image = self.transform(img)

        return image, self.labels[index]

class EmptyDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None ):
        self.images = [path]
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        img = Image.open(image_path)
        img = img.convert('RGB')

        if self.transform is not None:
            image = self.transform(img)
            
        return image
