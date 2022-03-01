import os
import sys

from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataset import GeneralDataset

GPU = 0
SIZE = 256
BS = 32
NUMCLS = 2
LR = 1e-4
WEIGHTDECAY=1e-4
EPOCHS = 15
PATH = r"C:\Users\esuh\Downloads\train\train"

def set_loader():
    ##----------Transforms----------##
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    scale = (0.875, 1.)

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=SIZE, scale= scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    # test_transform = transforms.Compose([
    #     transforms.Resize(SIZE),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    ##----------Dataset----------##
    
    train_dataset = GeneralDataset(PATH,transform=train_transform)
    # test_dataset = MVTECDataset(test_names_T,transform=test_transform)
    
    ##----------Dataloader----------##
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BS, shuffle= True,
        num_workers=12, pin_memory=True, sampler=None, drop_last = True)
        
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=opt.batch_size, shuffle= False,
    #     num_workers=opt.num_workers, pin_memory=True, sampler=None)

    return train_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_model(model, optimizer, epoch, save_path):
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_path)
    del state


def trainer(trainloader, model, criterion, optimizer, epoch):
    """one epoch training"""
    model.train()
    losses_CE = AverageMeter()
    top1 = AverageMeter()

    for idx, samples in enumerate(trainloader):
        images, labels = samples

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # compute loss
        output = model(images)
        loss_CE = criterion(output, labels)
            
        # update metric
        losses_CE.update(loss_CE.item(), BS)
        acc1, _ = accuracy(output[:BS,:], labels[:BS], topk=(1, 2))
        top1.update(acc1[0], BS)

        # SGD
        optimizer.zero_grad()
        total_loss = loss_CE
        total_loss.backward()
        optimizer.step()

    return losses_CE.avg, top1.avg
 
def main():

    loader = set_loader()

    # build model and criterion
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, NUMCLS)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # build optimizer
    optimizer = optim.Adam(model.parameters(),
                        lr=LR,
                        weight_decay=WEIGHTDECAY)

    # training routine
    for epoch in range(1, EPOCHS + 1):
        loss, train_acc = trainer(loader, model, criterion, optimizer, epoch)
        print(f'epoch {epoch}/{EPOCHS}, loss: {loss}, accuracy: {train_acc}')
    
    save_path = os.path.join(
        './models', 'last.pth')
    save_model(model, optimizer, epoch, save_path)

if __name__ == '__main__':
    main()
