from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, L1Loss, SmoothL1Loss
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.linalg import norm

def laplacian_eigenvector_loss(pe: Tensor, adj_mat: Tensor, batch_index: Tensor, lambda_loss: float) -> Tensor:
    '''
        Loss fn for positional encoding
        https://arxiv.org/pdf/2110.07875.pdf
        Src: https://github.com/vijaydwivedi75/gnn-lspe/blob/main/nets/ZINC_graph_regression/gatedgcn_net.py#L132
    '''
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

    loss_pe = (loss_pe_1 + lambda_loss * loss_pe_2) / (k * batch_size * n) 

    return loss_pe

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

def std_l1_loss(input: Tensor, target: Tensor, reduction: str='mean'):
    std = torch.std(target, dim=-1)
        
    loss = F.l1_loss(input, target, reduction='none')

    if reduction == 'none':
        return (loss.T / std).T
    else:
        if len(loss.shape) > 1:
            loss = torch.mean(loss, dim=-1)

        if reduction == 'mean':
            return torch.mean(loss / std)
        elif reduction == 'sum':
            return torch.sum(loss / std)

class StdL1Loss(Module):
    '''
        Src: http://cjcp.ustc.edu.cn/hxwlxb/article/doi/10.1063/1674-0068/cjcp2203055
    '''
    def __init__(self, reduction: str='mean'):
        super(StdL1Loss, self).__init__()
        self.l1_loss = L1Loss(reduction='none')
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor):
        std = torch.std(target, dim=-1)
        
        loss = self.l1_loss(input, target)

        if self.reduction == 'none':
            return (loss.T / std).T
        else:
            if len(loss.shape) > 1:
                loss = torch.mean(loss, dim=-1)

            if self.reduction == 'mean':
                return torch.mean(loss / std)
            elif self.reduction == 'sum':
                return torch.sum(loss / std)

def std_smooth_l1_loss(input: Tensor, target: Tensor, reduction: str='mean'):
    std = torch.std(target, dim=-1)
        
    loss = F.smooth_l1_loss(input, target, reduction='none')

    if reduction == 'none':
        return (loss.T / std).T
    else:
        if len(loss.shape) > 1:
            loss = torch.mean(loss, dim=-1)

        if reduction == 'mean':
            return torch.mean(loss / std)
        elif reduction == 'sum':
            return torch.sum(loss / std)

class StdSmoothL1Loss(Module):
    '''
        Src: http://cjcp.ustc.edu.cn/hxwlxb/article/doi/10.1063/1674-0068/cjcp2203055
    '''
    def __init__(self, reduction: str='mean'):
        super(StdSmoothL1Loss, self).__init__()
        self.l1_loss = SmoothL1Loss(reduction='none')
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor):
        std = torch.std(target, dim=-1)
        
        loss = self.l1_loss(input, target)

        if self.reduction == 'none':
            return (loss.T / std).T
        else:
            if len(loss.shape) > 1:
                loss = torch.mean(loss, dim=-1)

            if self.reduction == 'mean':
                return torch.mean(loss / std)
            elif self.reduction == 'sum':
                return torch.sum(loss / std)

class UncertaintyLoss(Module):
    '''
        Revised Uncertainty Loss function
        Src: http://cjcp.ustc.edu.cn/hxwlxb/article/doi/10.1063/1674-0068/cjcp2203055
             https://arxiv.org/abs/1805.06334
    '''
    def __init__(self, num_tasks: int):
        super(UncertaintyLoss, self).__init__()

        self.num_tasks = num_tasks
        self.log_sigma = Parameter(torch.zeros(num_tasks))

    def forward(self, loss: Tensor):
        print('log_sigma', self.log_sigma)

        if self.num_tasks == 1:
            return torch.mean(loss)

        loss_l = 0.5 * loss / torch.exp(self.log_sigma) ** 2
        loss_r = torch.log(1 + torch.exp(self.log_sigma) ** 2)

        return torch.sum(loss_l + loss_r)

class DynamicWeightAverageLoss(Module):
    '''
        Src: http://cjcp.ustc.edu.cn/hxwlxb/article/doi/10.1063/1674-0068/cjcp2203055
             https://arxiv.org/abs/1803.10704
    '''
    def __init__(self, num_tasks: int):
        super(DynamicWeightAverageLoss, self).__init__()

        self.num_tasks = num_tasks

        # temperature
        self.T = 2

        self.prev_prev_loss = torch.empty(0)
        self.prev_loss = torch.empty(0)

        self.weight = torch.tensor([1.] * self.num_tasks, requires_grad=False)

    def get_last_delta_loss(self) -> Tensor:
        return self.prev_loss / self.prev_prev_loss

    def forward(self, loss: Tensor, iteration: int=None):
        print('weight', self.weight)

        if iteration == None:
            return torch.sum(self.weight * loss)

        if iteration == 1 or iteration == 2:
            w = torch.tensor([1.] * self.num_tasks, requires_grad=False)
        else:
            w = self.get_last_delta_loss()

        # update
        self.prev_prev_loss = self.prev_loss.clone().detach()
        self.prev_loss = loss.clone().detach()

        e = torch.exp(w / self.T)

        self.weight = self.num_tasks * e / e.sum()
        self.weight = self.weight.cuda()

        return torch.sum(self.weight * loss)