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
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge

from nets import FC, Conv_Net, ResNet18, VGG, Conv_Net_Mnist
from ntk import kernel_mats_d_gan, kernel_svm, kernel_ridge, kernel_mats_d_gan_manifold

parser = argparse.ArgumentParser()

parser.add_argument("--dataset",
                    type=str,
                    default="cifar10",
                    choices=["mnist", "cifar10", "curves"])
parser.add_argument("--nn_architecture",
                    type=str,
                    default="Conv",
                    choices=["FC", "Conv", "Res", "VGG"])
parser.add_argument('--tb-number', type=int, default=3)
parser.add_argument('--tb-path', type=str, default='tb-cifar10-conv')
parser.add_argument('--chk', type=str, default='chks')
parser.add_argument("--pos_class",type=int,default=5)
parser.add_argument("--neg_class",type=int,default=3)

args = parser.parse_args()

# specify datasets, network architecture, and pos & neg class index
args.dataset = "curves"
# args.dataset = "cifar10"
# args.dataset = "mnist"
# args.nn_architecture = "Conv"
args.nn_architecture = "FC"
args.tb_number = 0
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


img_channel = 1
if args.dataset=="cifar10":
    img_channel = 3

if args.dataset=="cifar10":
    n_0 = 3072
elif args.dataset=="mnist":
    n_0 = 28*28
n = 300
L = 5

states = torch.load(os.path.join(chk_path, 'last_epoch.pt'))
epochs = states['epochs']
losses = states['losses']
x = states['x']
y = states['y']

net_errs = []
ntks = []

svm_errs = []
krr_errs = []

images = x
labels = y
targets = labels.detach().numpy().flatten()


# for epoch in range(100):  # loop over the dataset multiple times
#     if epoch % 10 == 9:    # print every 10 epochs
for epoch in epochs:
      start_time = time.time()
      print("computing NTK of epoch: {}".format(epoch))
      if args.nn_architecture == "Conv":
          if args.dataset == "cifar10":
              net = Conv_Net(img_channel)
          elif args.dataset == "mnist":
              net = Conv_Net_Mnist()
      elif args.nn_architecture == "FC":
          net = FC(n_0, n, L)
      elif args.nn_architecture == "Res":
          net = ResNet18()
      else:
          net = VGG('VGG11')
      net.load_state_dict(torch.load(chk_path+'/'+str(epoch)+'.pth'))
      net_error = net(images).data - labels
      net_errs.append(net_error)
      ntk = kernel_mats_d_gan_manifold(net, images, labels, use_cuda=False, kernels='trainvtrain')
      ntks.append(ntk)
      print("compute time: {}".format(time.time() - start_time))
      # svm_pred = kernel_svm(ntk, targets);
      # krr_pred = kernel_ridge(ntk, targets);
      # svm_errs.append(np.square(svm_pred - targets).mean())
      # krr_errs.append(np.square(krr_pred - targets).mean())


out_dict = {'epochs': epochs,
            'net_errs': net_errs,
            'ntks': ntks,
            'labels': labels,
            'losses': losses}
torch.save(out_dict, os.path.join(chk_path, 'ntks.pt'))

