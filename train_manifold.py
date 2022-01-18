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
args.dataset = "curves"
# args.nn_architecture = "Conv"
args.nn_architecture = "FC"
args.tb_number = 0
args.momentum = 0
args.maxepoch = 100
args.lr = 0.1
img_channel = 1
N = 100
n_0 = 1000
n = 300
L = 5

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

def generate_data_coaxial_circle(n_0, N, r=1):
  x = np.zeros((2*N, n_0))
  for i in np.arange(N):
     x[i, 0] = r*math.cos(2.0*math.pi*i/N)
     x[i, 1] = r*math.sin(2.0*math.pi*i/N)
     x[i, 2] = np.sqrt(1-r**2)
  for i in np.arange(N):
     x[N+i, 0] = r*math.cos(2.0*math.pi*i/N)
     x[N+i, 1] = r*math.sin(2.0*math.pi*i/N)
     x[N+i, 2] = -np.sqrt(1-r**2)
  u = np.random.rand(n_0, n_0)
  q, r = np.linalg.qr(u)
  x = np.matmul(x, q)
  x = torch.from_numpy(x).float()
  return x


def generate_data_sine_curves(n_0, N, a0 = math.pi/16, offset = 0.25, omega0 = 10, theta0=2/math.pi, theta1=16 / math.pi):
    x = np.zeros((2 * N, n_0))
    a0 = math.pi/16
    offset = 0.25
    t = np.linspace(0, math.pi, N)
    phi = np.linspace(0, 2 * math.pi, N)
    omega0 = 10
    theta0 = math.pi/2
    theta1 = (theta0 - offset) + a0 * np.sin(omega0 * phi)
    x[:N, 0] = (np.sin(theta1) * np.cos(phi))
    x[:N, 1] = (np.sin(theta1) * np.sin(phi))
    x[:N, 2] = np.cos(theta1)

    theta2 = (theta0 + offset) + a0 * np.sin(omega0 * phi)
    x[N:, 0] = np.sin(theta2) * np.cos(phi)
    x[N:, 1] = np.sin(theta2) * np.sin(phi)
    x[N:, 2] = np.cos(theta2)

    u = np.random.rand(n_0, n_0)
    q, r = np.linalg.qr(u)
    x = np.matmul(x, q)
    x = torch.from_numpy(x).float()

    return x

N_p = N
N_n = N
x = generate_data_sine_curves(n_0, N)
y_p = torch.ones(N_p, 1)
y_m = -torch.ones(N_n, 1)
y = torch.cat((y_p, y_m))

# net = Two_Layer_Net(n_0, n)
net = FC(n_0, n, L)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

losses = []
start_time = time.time()
print("start training")
for epoch in range(args.maxepoch):  # loop over the dataset multiple times

    running_loss = 0.0
    loss_count = 0
    for i in np.arange(np.floor(N/2)):
        # get the inputs; data is a list of [inputs, labels]
        inputs = x[(4*i).astype(int):(4*i+4).astype(int), :]
        labels = y[(4*i).astype(int):(4*i+4).astype(int)]
        labels = labels.to(torch.float32)
        # as we are using dogs and cats, transform the target into +1 and -1.
        # labels = torch.unsqueeze(labels-4, 1)
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
            'x': x,
            'y': y,
            'losses': losses,
               'net_state_dict': net.state_dict()}
torch.save(out_dict, os.path.join(chk_path, 'last_epoch.pt'))
print('Finished Training')

# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in trainloader:
#         images, labels = data
#         labels = torch.unsqueeze(labels-4, 1)
#         # images = images.view(images.size(0), -1)
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         predicted = torch.sign(outputs.data)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print(correct, total)
# print('Accuracy of the network on the 2000 test images: %d %%' % (
#     100 * correct / total))
#
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         labels = torch.unsqueeze(labels-4, 1)
#         # images = images.view(images.size(0), -1)
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         predicted = torch.sign(outputs.data)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print(correct, total)
# print('Accuracy of the network on the 2000 test images: %d %%' % (
#     100 * correct / total))


