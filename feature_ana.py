import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys

import gc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from scipy.sparse.linalg import svds

def feature_analysis(graphs, features, model, criterion_summed, weight_decay, device, C, loader):
    # register hook that saves last-layer input into features
    model.eval()

    N = [0 for _ in range(C)]
    mean = [0 for _ in range(C)]
    Sw = 0

    loss = 0
    net_correct = 0
    NCC_match_net = 0

    for computation in ['Mean', 'Cov']:
        for batch_idx, (data, target) in enumerate(loader, start=1):

            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output = model(data)
            #features_pool = F.avg_pool2d(features.value, 4)
            #h = features_pool.view(data.shape[0], -1)  # B CHW
            h = features.value.data.view(data.shape[0], -1)  # B CHW


            # during calculation of class means, calculate loss
            if computation == 'Mean':
                if str(criterion_summed) == 'CrossEntropyLoss()':
                    loss += criterion_summed(output, target).item()
                elif str(criterion_summed) == 'MSELoss()':
                    loss += criterion_summed(output, F.one_hot(target, num_classes=num_classes).float()).item()

            for c in range(C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple=True)[0]

                if len(idxs) == 0:  # If no class-c in this batch
                    continue
                h_c = h[idxs, :]  # B CHW
                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0)  #  CHW
                    N[c] += h_c.shape[0]

                elif computation == 'Cov':
                    # update within-class cov

                    z = h_c - mean[c].unsqueeze(0)  # B CHW
                    cov = torch.matmul(z.unsqueeze(-1),  # B CHW 1
                                       z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim=0)

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs, :], dim=1)
                    net_correct += sum(net_pred == target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i, :] - M.T, dim=1) \
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                    NCC_match_net += sum(NCC_pred == net_pred).item()

        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)

    graphs.loss.append(loss)
    graphs.accuracy.append(net_correct / sum(N))
    graphs.NCC_mismatch.append(1 - NCC_match_net / sum(N))

    # loss with weight decay
    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * weight_decay * torch.sum(param ** 2).item()
    graphs.reg_loss.append(reg_loss)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True)  # CHW 1

    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C  # conv4: 16384 * 16384

    # avg norm
    # W = classifier.weight
    M_norms = torch.norm(M_, dim=0)
    # W_norms = torch.norm(W.T, dim=0)

    graphs.norm_M_CoV.append((torch.std(M_norms) / torch.mean(M_norms)).item())
    # graphs.norm_W_CoV.append((torch.std(W_norms) / torch.mean(W_norms)).item())

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=C - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb))

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_, 'fro')
    # normalized_W = W.T / torch.norm(W.T, 'fro')
    #graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M) ** 2).item())

    # mutual coherence
    def coherence(V):
        G = V.T @ V
        G += torch.ones((C, C), device=device) / (C - 1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G, 1).item() / (C * (C - 1))

    graphs.cos_M.append(coherence(M_ / M_norms))
    #graphs.cos_W.append(coherence(W.T / W_norms))
    return graphs
