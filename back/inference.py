import os
import sys

from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataset import EmptyDataset

GPU = 0
SIZE = 256
NUMCLS = 2

def set_loader(path):
    ##----------Transforms----------##
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    ##----------Dataset----------##    
    test_dataset = EmptyDataset(path,transform=test_transform)
    
    ##----------Dataloader----------##
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle= False,
        num_workers=1, sampler=None)

    return test_loader

def tester(test_loader, model):
    model.eval()

    with torch.no_grad():
        for idx, (images) in enumerate(test_loader):
            # forward
            output = model(images)
            prob = torch.nn.functional.softmax(output, dim=1)[:,1]
    return prob

async def _inference(path):

    loader = set_loader(path)

    # build model and criterion
    model = models.resnet18()
    model.fc = nn.Linear(512, NUMCLS)
    pretrained_dict = torch.load('./models/last.pth', map_location=torch.device('cpu'))['model']
    model.load_state_dict(pretrained_dict)

    # inference routine
    prediction = tester(loader, model)
    prediction = prediction[0].item()
    return round(1-prediction,4), round(prediction,4)
    
PATH = r"C:\Users\esuh\Downloads\test1\test1\8.jpg"
if __name__ == '__main__':
    print(_inference(PATH))