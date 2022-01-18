import numpy as np
import torch

def compute_error_norm_dyn(kernel, init_err, n_rpt=100, step_size=1e-3):
    eig_err_align_norms = []
    err_norms = []
    err = init_err

    L, V = torch.linalg.eig(kernel)
    V = V.real
    err_norms.append(torch.square(err).mean())
    eig_err_align = torch.matmul(V.T, err)
    eig_err_align_norm = np.zeros_like(eig_err_align)
    for i in np.arange(len(eig_err_align)):
        eig_err_align_norm[i] = np.linalg.norm(eig_err_align[:i])

    for rpt in np.arange(n_rpt):
        err = err - step_size * torch.matmul(kernel, err)