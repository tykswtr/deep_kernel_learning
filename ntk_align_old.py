import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import math
import argparse
import os
import sys
# from logger import Logger

import torchvision
import torchvision.transforms as transforms

import sklearn
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge

from nets import FC, Conv_Net
from ntk import kernel_mats_d_gan, kernel_svm, kernel_ridge

parser = argparse.ArgumentParser()

parser.add_argument("--dataset",
                    type=str,
                    default="cifar10",
                    choices=["mnist", "cifar10", "curves"])
parser.add_argument("--nn_architecture",
                    type=str,
                    default="Conv",
                    choices=["FC", "Conv", "Res"])
parser.add_argument('--tb-number', type=int, default=0)
parser.add_argument('--tb-path', type=str, default='tb-cifar10-conv')
parser.add_argument('--chk', type=str, default='chks/cifar10-conv')
parser.add_argument("--pos_class",type=int,default=5)
parser.add_argument("--neg_class",type=int,default=3)

args = parser.parse_args()
print(args)
sys.stdout.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("working over: {}".format(device))

chk_path = os.path.join(args.chk, str(args.pos_class)+str(args.neg_class), str(args.tb_number))
if not os.path.exists(chk_path):
    os.makedirs(chk_path)


if args.dataset=="cifar10":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # only getting a subset of cifar10 dataset
    from torch.utils.data import Subset

    pos_class = 'dog'
    neg_class = 'cat'
    N = 100

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

fb_trainloader = torch.utils.data.DataLoader(binary_train_dataset, batch_size=len(binary_train_dataset),
                                          shuffle=False, num_workers=0)

states = torch.load(os.path.join(chk_path, 'last_epoch.pt'))
print(states['epoch'])
losses = states['loss']

svm_errs = [];
krr_errs = [];

images = [];
labels = [];

for data in fb_trainloader:
    images, labels = data
    labels = torch.unsqueeze(labels-4, 1)
    labels = labels.to(torch.float32)
    targets = labels.detach().numpy().flatten()

epochs = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
# for epoch in range(100):  # loop over the dataset multiple times
#     if epoch % 10 == 9:    # print every 10 epochs
for epoch in epochs:
      net = Conv_Net()
      net.load_state_dict(torch.load(chk_path+'/'+str(epoch)+'.pth'))
      net_error = net(images).data - labels
      ntk = kernel_mats_d_gan(net, binary_train_dataset, binary_test_dataset, use_cuda = False, kernels='trainvtrain')
      svm_pred = kernel_svm(ntk, targets);
      krr_pred = kernel_ridge(ntk, targets);
      svm_errs.append(np.square(svm_pred - targets).mean())
      krr_errs.append(np.square(krr_pred - targets).mean())

plt.figure()
# plt.subplot(1, 2, 1)
plt.plot(epochs, svm_errs, label = 'svm errs')
plt.plot(epochs, krr_errs, label = 'krr errs')
# plt.legend()
# plt.subplot(1,2,2)
plt.plot(epochs, losses, label = 'training loss')
plt.legend()
plt.savefig(os.path.join(chk_path, 'kernel_err.png'))

fb_trainloader = torch.utils.data.DataLoader(binary_train_dataset, batch_size=len(binary_train_dataset),
                                          shuffle=False, num_workers=0)

cert_targets = [];
cert_errs = [];
eig_target_align_norms = [];
eig_err_align_norms = [];
err_norms = [];

images = [];
labels = [];

for data in fb_trainloader:
    images, labels = data
    labels = torch.unsqueeze(labels-4, 1)
    labels = labels.to(torch.float32)
    targets = labels.detach().numpy().flatten()

epochs = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
# for epoch in range(100):  # loop over the dataset multiple times
#     if epoch % 10 == 9:    # print every 10 epochs
for epoch in epochs:
      net = Conv_Net()
      net.load_state_dict(torch.load(chk_path+'/'+str(epoch)+'.pth'))
      net_error = net(images).data - labels
      err_norms.append(torch.square(net_error).mean())
      ntk = kernel_mats_d_gan(net, binary_train_dataset, binary_test_dataset, use_cuda = False, kernels='trainvtrain')
      # g_target = torch.linalg.solve(ntk, labels.detach())
      # g_err = torch.linalg.solve(ntk, net_error.detach())
      # ntk_inv = torch.linalg.inv(ntk+1e-5*torch.eye(len(labels)))
      ntk_inv = torch.inverse(ntk + 1e-5 * torch.eye(len(labels)))
      g_target = torch.matmul(ntk_inv, labels)
      g_err = torch.matmul(ntk_inv, net_error)
      # print(torch.allclose(ntk @ g_target, labels))
      # print(torch.allclose(ntk @ g_err, net_error))
      cert_targets.append(torch.linalg.norm(g_target))
      cert_errs.append(torch.linalg.norm(g_err))
      # L, V = torch.linalg.eig(ntk)
      # V = V.real
      L, V = torch.eig(ntk, eigenvectors=True)
      eig_target_align = torch.matmul(V.T, labels)
      eig_err_align = torch.matmul(V.T, net_error)
      eig_target_align_norm = np.zeros_like(eig_target_align);
      eig_err_align_norm = np.zeros_like(eig_err_align);
      for i in np.arange(len(eig_target_align)):
        eig_target_align_norm[i] = np.linalg.norm(eig_target_align[:i]);
        eig_err_align_norm[i] = np.linalg.norm(eig_err_align[:i]);
      eig_target_align_norms.append(eig_target_align_norm)
      eig_err_align_norms.append(eig_err_align_norm)


plt.figure()
# plt.plot(epochs, cert_targets, label='target cert')
plt.plot(epochs, [a / b for a, b in zip(cert_errs, err_norms)], label='err cert')
plt.legend()
plt.savefig(os.path.join(chk_path, 'err_cert_norm.png'))
plt.figure()
plt.plot(epochs, [a / b for a, b in zip(cert_targets, err_norms)], label='target cert')
# plt.plot(epochs, cert_errs, label='err cert')
plt.legend()
plt.savefig(os.path.join(chk_path, 'target_cert_norm.png'))

plt.figure()
for i, v in enumerate(eig_target_align_norms):
  plt.plot(v, label='epoch '+str(epochs[i]))
plt.title('NTK alignment with labels')
plt.legend()
plt.savefig(os.path.join(chk_path, 'target_eig_align.png'))
plt.figure()
for i, v in enumerate(eig_err_align_norms):
  plt.plot(v/torch.sqrt(err_norms[i]), label='epoch '+str(epochs[i]))
plt.legend()
plt.title('NTK alignment with errors')
plt.savefig(os.path.join(chk_path, 'err_eig_align.png'))

