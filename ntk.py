# Borrowing kernel computation code from https://github.com/bobby-he/Neural_Tangent_Kernel
import numpy as np
import torch
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge

def kernel_mats_d_gan(net, d_train, d_test, use_cuda=True, kernels='both'):
    # for a given net, this function computes the K_testvtrain (n_test by n_train) and the
    # K_trainvtrain (n_train by n_train) kernels.
    # You can choose which one to return by the parameter 'kernels', with values 'both' (default), 'testvtrain' or 'trainvtrain'

    # suppose cuda available
    n_test_pts = len(d_test)
    n_train_pts = len(d_train)
    if use_cuda:
        net = net.cuda()
    # the following computes the gradients with respect to all parameters
    grad_list = []

    kernel_trainloader = torch.utils.data.DataLoader(d_train, batch_size=1,
                                                     shuffle=False, num_workers=0)
    for i, data in enumerate(kernel_trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.to(torch.float32)
        # for d_point in d_train:
        #     if use_cuda:
        #         d_point = d_point.cuda()
        loss = net(inputs)
        grad_list.append(torch.autograd.grad(loss, net.parameters(), retain_graph=True))

    # testvstrain kernel
    if kernels == 'both' or kernels == 'testvtrain':
        K_testvtrain = torch.zeros((n_test_pts, n_train_pts))
        for i, d_point in enumerate(d_test):
            # if ((i+1)*10)%len(d_test) == 0:
            # print('K_testvtrain is {}% complete'.format(int((i+1)/len(d_test)*100)))
            if use_cuda:
                d_point = d_point.cuda()
            loss = net(d_point)
            grads = torch.autograd.grad(loss, net.parameters(), retain_graph=True)  # extract NN gradients
            for j in range(len(grad_list)):
                pt_grad = grad_list[j]  # the gradients at the jth (out of 4) data point
                K_testvtrain[i, j] = sum([torch.sum(torch.mul(grads[u], pt_grad[u])) for u in range(len(grads))])

    # trainvstrain kernel
    if kernels == 'both' or kernels == 'trainvtrain':
        K_trainvtrain = torch.zeros((n_train_pts, n_train_pts))
        for i in range(len(grad_list)):
            grad_i = grad_list[i]
            for j in range(i + 1):
                grad_j = grad_list[j]
                K_trainvtrain[i, j] = sum([torch.sum(torch.mul(grad_i[u], grad_j[u])) for u in range(len(grad_j))])
                K_trainvtrain[j, i] = K_trainvtrain[i, j]

    if kernels == 'both':
        return K_testvtrain, K_trainvtrain
    elif kernels == 'trainvtrain':
        return K_trainvtrain
    elif kernels == 'testvtrain':
        return K_testvtrain

def kernel_mats_d_gan_manifold(net, inputs, labels, use_cuda=True, kernels='both'):
    # for a given net, this function computes the K_testvtrain (n_test by n_train) and the
    # K_trainvtrain (n_train by n_train) kernels.
    # You can choose which one to return by the parameter 'kernels', with values 'both' (default), 'testvtrain' or 'trainvtrain'

    # suppose cuda available
    n_train_pts = len(labels)
    if use_cuda:
        net = net.cuda()
    # the following computes the gradients with respect to all parameters
    grad_list = []
    for i in np.arange(n_train_pts):
        loss = net(torch.unsqueeze(inputs[i, :], 0))
        grad_list.append(torch.autograd.grad(loss, net.parameters(), retain_graph=True))

    # trainvstrain kernel
    if kernels == 'both' or kernels == 'trainvtrain':
        K_trainvtrain = torch.zeros((n_train_pts, n_train_pts))
        for i in range(len(grad_list)):
            grad_i = grad_list[i]
            for j in range(i + 1):
                grad_j = grad_list[j]
                K_trainvtrain[i, j] = sum([torch.sum(torch.mul(grad_i[u], grad_j[u])) for u in range(len(grad_j))])
                K_trainvtrain[j, i] = K_trainvtrain[i, j]

    return K_trainvtrain


def kernel_svm(ntk, y):
  clf = svm.SVR(kernel='precomputed')
  clf.fit(ntk, y)
  return clf.predict(ntk)

def kernel_ridge(ntk, y):
  krr = KernelRidge(alpha=1.0,kernel='precomputed')
  krr.fit(ntk, y)
  return krr.predict(ntk)