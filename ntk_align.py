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
import time
# from logger import Logger

import torchvision
import torchvision.transforms as transforms

import sklearn
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge

from nets import FC, Conv_Net, ResNet18, VGG, Conv_Net_Mnist
from ntk import kernel_mats_d_gan, kernel_svm, kernel_ridge

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
# args.dataset = "curves"
# args.dataset = "cifar10"
args.dataset = "mnist"
# args.nn_architecture = "Res"
args.nn_architecture = "Conv"
# args.nn_architecture = "FC"
args.tb_number = 6
# N = 100

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

states = torch.load(os.path.join(chk_path, 'ntks.pt'))
epochs = states['epochs']
net_errs = states['net_errs']
ntks = states['ntks']
labels = states['labels']
losses = states['losses']
targets = labels.detach().numpy().flatten()

svm_errs = []
krr_errs = []

# for epoch in range(100):  # loop over the dataset multiple times
#     if epoch % 10 == 9:    # print every 10 epochs
start_time = time.time()
print("ntk size: {}".format(ntks[0].size()))
print("compute kernel regression")
for epoch_id, epoch in enumerate(epochs):
      net_error = net_errs[epoch_id]
      ntk = ntks[epoch_id]
      svm_pred = kernel_svm(ntk, targets);
      krr_pred = kernel_ridge(ntk, targets);
      svm_errs.append(np.square(svm_pred - targets).mean())
      krr_errs.append(np.square(krr_pred - targets).mean())
print("finish in time: {}".format(time.time() - start_time))

plt.figure()
# plt.subplot(1, 2, 1)
plt.plot(epochs, svm_errs, label = 'svm errs')
plt.plot(epochs, krr_errs, label = 'krr errs')
# plt.legend()
# plt.subplot(1,2,2)
plt.plot(epochs, losses, label = 'training loss')
plt.legend()
plt.savefig(os.path.join(chk_path, 'kernel_err.png'))


cert_targets = []
cert_errs = []
eig_target_align_norms = []
eig_err_align_norms = []
err_norms = []
zeta_dtheta_zetas = []

img_channel = 1
if args.dataset=="cifar10":
    img_channel = 3

if args.dataset=="cifar10":
    n_0 = 3072
elif args.dataset=="mnist":
    n_0 = 28*28
n = 300
L = 5

# for epoch in range(100):  # loop over the dataset multiple times
#     if epoch % 10 == 9:    # print every 10 epochs
for epoch_id, epoch in enumerate(epochs):
      start_time = time.time()
      print("compute kernel alignment with epoch: {}".format(epoch))
      # if args.nn_architecture == "Conv":
      #     if args.dataset == "cifar10":
      #         net = Conv_Net(img_channel)
      #     elif args.dataset == "mnist":
      #         net = Conv_Net_Mnist()
      # elif args.nn_architecture == "FC":
      #     net = FC(n_0, n, L)
      # elif args.nn_architecture == "Res":
      #     net = ResNet18()
      # else:
      #     net = VGG('VGG11')
      # net.load_state_dict(torch.load(chk_path+'/'+str(epoch)+'.pth'))
      net_error = net_errs[epoch_id]
      err_norms.append(torch.square(net_error).mean())
      ntk = ntks[epoch_id]
      if epoch_id < len(epochs) - 1:
          ntk_diff = ntks[epoch_id + 1] - ntks[epoch_id]
          zeta_dtheta_zeta = torch.matmul(net_error.T, torch.matmul(ntk_diff, net_error))
          zeta_dtheta_zetas.append(zeta_dtheta_zeta)
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
      L, V = torch.linalg.eig(ntk)
      V = V.real
      # L, V = torch.eig(ntk, eigenvectors=True)
      eig_target_align = torch.matmul(V.T, labels)
      eig_err_align = torch.matmul(V.T, net_error)
      eig_target_align_norm = np.zeros_like(eig_target_align);
      eig_err_align_norm = np.zeros_like(eig_err_align);
      for i in np.arange(len(eig_target_align)):
        eig_target_align_norm[i] = np.linalg.norm(eig_target_align[:i]);
        eig_err_align_norm[i] = np.linalg.norm(eig_err_align[:i]);
      eig_target_align_norms.append(eig_target_align_norm)
      eig_err_align_norms.append(eig_err_align_norm)
      print("compute time: {}".format(time.time() - start_time))


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

plt.figure()
# plt.plot(epochs, cert_targets, label='target cert')
plt.plot(epochs[:-1], zeta_dtheta_zetas, label='zeta dTheta zeta')
plt.legend()
plt.savefig(os.path.join(chk_path, 'zeta_dTheta_zeta.png'))


init_ntk = ntks[0]
init_ntk_inv = torch.inverse(init_ntk + 1e-5 * torch.eye(len(labels)))
net_error = net_errs[0]
L, V = torch.linalg.eig(init_ntk)
V = V.real
init_eig_err_align_norms = []
init_err_norms = []
kernel_cert_errs = []
kernel_cert_targets = []
# step_size = 1e-1;
step_size = 1e-5
n_rpt = 50000

for epoch in np.arange(len(epochs)):
  init_err_norms.append(torch.square(net_error).mean())
  init_eig_err_align = torch.matmul(V.T, net_error)
  init_eig_err_align_norm = np.zeros_like(init_eig_err_align);
  for i in np.arange(len(init_eig_err_align)):
        init_eig_err_align_norm[i] = np.linalg.norm(init_eig_err_align[:i]);
  init_eig_err_align_norms.append(init_eig_err_align_norm)
  g_target = torch.matmul(init_ntk_inv, labels)
  g_err = torch.matmul(init_ntk_inv, net_error)
  kernel_cert_errs.append(torch.linalg.norm(g_err))
  kernel_cert_targets.append(torch.linalg.norm(g_target))
  for rpt in np.arange(n_rpt):
    net_error = net_error - step_size * torch.matmul(init_ntk, net_error)
plt.figure()
for i, v in enumerate(init_eig_err_align_norms):
  plt.plot(v/torch.sqrt(init_err_norms[i]), label='epoch '+str(epochs[i]))
plt.legend()
plt.title('initial NTK alignment with errors under kernel dynamics')
plt.savefig(os.path.join(chk_path, 'alignment_kernel_dyn.png'))

plt.figure()
# plt.plot(epochs, cert_targets, label='target cert')
plt.plot(epochs, [a / b for a, b in zip(kernel_cert_errs, err_norms)], label='err cert')
plt.legend()
plt.savefig(os.path.join(chk_path, 'err_cert_norm_kernel_dyn.png'))
plt.figure()
plt.plot(epochs, [a / b for a, b in zip(kernel_cert_targets, err_norms)], label='target cert')
# plt.plot(epochs, cert_errs, label='err cert')
plt.legend()
plt.savefig(os.path.join(chk_path, 'target_cert_norm_kernel_dyn.png'))

