from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module
from torch.linalg import norm

class LaplacianEigenvectorLoss(Module):
    '''
        Loss fn for positional encoding
        https://arxiv.org/pdf/2110.07875.pdf
        Src: https://github.com/vijaydwivedi75/gnn-lspe/blob/main/nets/ZINC_graph_regression/gatedgcn_net.py#L132
    '''
    def __init__(self, lambda_loss: float=0.7):
        super(LaplacianEigenvectorLoss, self).__init__()
        # hyperparam
        self.lambda_loss = lambda_loss

    def forward(self, pe: Tensor, adj_mat: Tensor, batch_index: Tensor) -> Tensor:
        d = adj_mat.size(0) # feature length
        k = pe.size(-1) # walk length
        n = pe.size(0) # num_nodes in batch

        A = adj_mat.float()
        deg_A = torch.sum(A, dim=1, dtype=torch.float)
        mask = deg_A.eq(0.)
        deg_A = deg_A.masked_fill(mask, 1.)
        D = torch.diag(deg_A) # degree mat

        D_inv_sqrt = D.inverse() ** 0.5

        # normalized Laplacian matrix
        L = torch.eye(d, dtype=torch.float) - D_inv_sqrt @ A @ D_inv_sqrt
        L = L.cuda()

        loss_pe_1 = torch.trace(pe.T @ L @ pe)

        # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
        batch_size = len(batch_index) - 1
        P = torch.block_diag(*[pe[batch_index[i] : batch_index[i+1]] for i in range(batch_size)])

        PTP_inv = P.T @ P - torch.eye(P.size(-1)).cuda()

        # Frobenius norm
        loss_pe_2 = norm(PTP_inv, ord='fro') ** 2
        loss_pe_2 = loss_pe_2.cuda()

        loss_pe = (loss_pe_1 + self.lambda_loss * loss_pe_2) / (k * batch_size * n) 

        return loss_pe
