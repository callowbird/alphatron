import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F

import densenet as dn

# used for logging to TensorBoard

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=1.0, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='runs/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.bottleneck=False

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/home/yang/data/data/', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/home/yang/data/data/', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    model = dn.DenseNet3(args.layers, 10, args.growth, reduction=args.reduce,
                         bottleneck=args.bottleneck, dropRate=args.droprate)
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_data=extract_train(train_loader, model)

    val_data=extract_validate(val_loader, model)

    torch.save(train_data,"train.dat")
    torch.save(val_data,"val.dat")

def model_extract(model,x):
    out = model.conv1(x)
    out = model.trans1(model.block1(out))
    out = model.trans2(model.block2(out))
    out = model.block3(out)
    out = model.relu(model.bn1(out))
    out = F.avg_pool2d(out, 8)
    out = out.view(-1, model.in_planes)
    return out

def extract_train(train_loader, model):
    model.train()
    len_loader=50000
    gatherx=torch.Tensor(len_loader,456)
    gathery=torch.Tensor(len_loader)

    index_p=0

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)

        output = model_extract(model, input_var)

        if index_p+input.size(0)>len_loader:
            end_p=len_loader
        else:
            end_p=index_p+input.size(0)
        gatherx[index_p:end_p].copy_(output.data)
        gathery[index_p:end_p].copy_(target)
        index_p+=input.size(0)

    return {'x': gatherx, 'y': gathery}

def extract_validate(val_loader, model):
    model.eval()
    len_loader=10000
    gatherx=torch.Tensor(len_loader,456)
    gathery=torch.Tensor(len_loader)

    index_p=0

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)

        output = model_extract(model, input_var)

        if index_p+input.size(0)>len_loader:
            end_p=len_loader
        else:
            end_p=index_p+input.size(0)
        gatherx[index_p:end_p].copy_(output.data)
        gathery[index_p:end_p].copy_(target)
        index_p+=input.size(0)

    return {'x': gatherx, 'y': gathery}


if __name__ == '__main__':
    main()
