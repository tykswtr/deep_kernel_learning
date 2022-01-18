import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset

import matplotlib.pyplot as plt
import math
import argparse
import os
import sys
import time
# from logger import Logger

import torchvision
import torchvision.transforms as transforms

import sklearn

from nets import FC, Conv_Net, ResNet18, VGG, Conv_Net_Mnist

parser = argparse.ArgumentParser()

parser.add_argument("--dataset",
                    type=str,
                    default="cifar10",
                    choices=["mnist", "cifar10", "curves"])
parser.add_argument("--nn_architecture",
                    type=str,
                    default="Conv",
                    choices=["FC", "Conv", "Res", "VGG"])
parser.add_argument('--tb-number', type=int, default=0)
parser.add_argument('--tb-path', type=str, default='tb-cifar10-conv')
parser.add_argument('--chk', type=str, default='chks')
parser.add_argument("--pos-class",type=int,default=5)
parser.add_argument("--neg-class",type=int,default=3)
parser.add_argument("--momentum",type=float,default=0)
parser.add_argument("--batchsize",type=int,default=4)
parser.add_argument("--weight_decay",type=float,default=0)
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument("--maxepoch",type=int,default=100)

args = parser.parse_args()

# specify datasets, network architecture, and pos & neg class index
args.dataset = "cifar10"
# args.dataset = "mnist"
# args.nn_architecture = "Conv"
args.nn_architecture = "FC"
args.tb_number = 7
args.momentum = 0
args.maxepoch = 100
N = 500

print(args)
sys.stdout.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("working over: {}".format(device))

# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# logger = Logger(os.path.join(args.tb_path, str(args.tb_number)))
chk_path = os.path.join(args.chk, args.dataset+"-"+args.nn_architecture, str(args.pos_class)+str(args.neg_class), str(args.tb_number))
if not os.path.exists(chk_path):
    os.makedirs(chk_path)

file1 = open(os.path.join(chk_path, "exp_setting.txt"),"w")
file1.write(str(args))
file1.close()

if args.dataset=="cifar10":

    # if args.nn_architecture=="Conv":
    #     transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # elif args.nn_architecture=="Res":
    #     transform_train = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # only getting a subset of cifar10 dataset
    from torch.utils.data import Subset

    pos_class = 'dog'
    neg_class = 'cat'

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    pos_indices, neg_indices, other_indices = [], [], []
    pos_idx, neg_idx = trainset.class_to_idx[pos_class], trainset.class_to_idx[neg_class]

    for i in range(len(trainset)):
        current_class = trainset[i][1]
        if current_class == pos_idx:
            pos_indices.append(i)
        elif current_class == neg_idx:
            neg_indices.append(i)
        else:
            other_indices.append(i)
    pos_indices = pos_indices[:N]
    neg_indices = neg_indices[:N]
    binary_train_dataset = Subset(trainset, pos_indices + neg_indices)

    batch_size = 4

    trainloader = torch.utils.data.DataLoader(binary_train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    pos_indices, neg_indices, other_indices = [], [], []
    for i in range(len(testset)):
        current_class = testset[i][1]
        if current_class == pos_idx:
            pos_indices.append(i)
        elif current_class == neg_idx:
            neg_indices.append(i)
        else:
            other_indices.append(i)
    # pos_indices = pos_indices[:N]
    # neg_indices = neg_indices[:N]
    binary_test_dataset = Subset(testset, pos_indices + neg_indices)

    testloader = torch.utils.data.DataLoader(binary_test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

if args.dataset=="mnist":

    transform = transforms.Compose(
        [transforms.ToTensor()])

    # only getting a subset of cifar10 dataset


    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    pos_indices, neg_indices, other_indices = [], [], []
    pos_idx, neg_idx = args.pos_class, args.neg_class

    for i in range(len(trainset)):
        current_class = trainset[i][1]
        if current_class == pos_idx:
            pos_indices.append(i)
        elif current_class == neg_idx:
            neg_indices.append(i)
        else:
            other_indices.append(i)
    pos_indices = pos_indices[:N]
    neg_indices = neg_indices[:N]
    binary_train_dataset = Subset(trainset, pos_indices + neg_indices)

    batch_size = 4

    trainloader = torch.utils.data.DataLoader(binary_train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    pos_indices, neg_indices, other_indices = [], [], []
    for i in range(len(testset)):
        current_class = testset[i][1]
        if current_class == pos_idx:
            pos_indices.append(i)
        elif current_class == neg_idx:
            neg_indices.append(i)
        else:
            other_indices.append(i)
    # pos_indices = pos_indices[:N]
    # neg_indices = neg_indices[:N]
    binary_test_dataset = Subset(testset, pos_indices + neg_indices)

    testloader = torch.utils.data.DataLoader(binary_test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

img_channel = 1
if args.dataset=="cifar10":
    img_channel = 3

n_0 = 3072
if args.dataset=="cifar10":
    n_0 = 3072
elif args.dataset=="mnist":
    n_0 = 28*28
n = 300
L = 5
# net = Two_Layer_Net(n_0, n)
# net = FC(n_0, n, L)
if args.nn_architecture=="Conv":
    if args.dataset=="cifar10":
        net = Conv_Net(img_channel)
    elif args.dataset=="mnist":
        net = Conv_Net_Mnist()
elif args.nn_architecture=="FC":
    net = FC(n_0, n, L)
elif args.nn_architecture=="Res":
    net = ResNet18()
else: net = VGG('VGG11')

# net = models.resnet18(pretrained=False)
# net.fc = nn.Linear(net.fc.in_features, 1)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

losses = []
start_time = time.time()
print("start training")
for epoch in range(args.maxepoch):  # loop over the dataset multiple times

    running_loss = 0.0
    loss_count = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.to(torch.float32)
        # as we are using dogs and cats, transform the target into +1 and -1.
        labels = torch.unsqueeze(labels-4, 1)
        # print(inputs.size())
        # inputs = inputs.view(inputs.size(0), -1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs)
        outputs = outputs.to(torch.float32)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        loss_count = loss_count + 1
    if epoch % 10 == 9:    # print every 10 epochs
      losses.append(running_loss / loss_count)
      torch.save(net.state_dict(), chk_path+'/'+str(epoch)+'.pth')
      print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i+1, running_loss / loss_count))
      print("compute time: {}".format(time.time() - start_time))
      start_time = time.time()
      running_loss = 0.0
      loss_count = 0

epochs = np.arange(9, args.maxepoch, 10)

out_dict = {'epochs': epochs,
            'losses': losses,
               'net_state_dict': net.state_dict()}
torch.save(out_dict, os.path.join(chk_path, 'last_epoch.pt'))
print('Finished Training')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        labels = torch.unsqueeze(labels-4, 1)
        # images = images.view(images.size(0), -1)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        predicted = torch.sign(outputs.data)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(correct, total)
print('Accuracy of the network on the 2000 test images: %d %%' % (
    100 * correct / total))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        labels = torch.unsqueeze(labels-4, 1)
        # images = images.view(images.size(0), -1)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        predicted = torch.sign(outputs.data)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(correct, total)
print('Accuracy of the network on the 2000 test images: %d %%' % (
    100 * correct / total))


