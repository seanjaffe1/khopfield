import torch
from torchvision import datasets, transforms

import torch

import torch.nn as nn

from lml import LML

class KHopfield(nn.Module):
    def __init__(self, N, n, p=1):
        # N is number of memories
        # n is dimension of memories
        super().__init__()
        self.N = N
        self.n = n
        self.memories = nn.Parameter(torch.randn(N, n,) * .1, requires_grad=True)
        # register memories
        self.p = p
    

    def to(self, device):
        super().to(device)
        self.memories = self.memories.to(device)
        return self
    
    def set_memories(self, memories):
        self.memories = torch.nn.Parameter(memories)

    def set_memories_from_loader(self, loader, max_batches=99999):
        memories = []
        for batch in loader:
            memories.append(batch[0])
            if len(memories) >= max_batches:
                break
        memories = torch.cat(memories)
        self.set_memories(memories)

    def forward(self, x, k, beta=1, hopfield_steps=0, index=False):
        # x: (batch_size,  n)\
        if index:
            return self.k_soft_hopfield_index(self.memories, x, k=k, beta=beta, hopfield_steps=hopfield_steps)
        return self.k_hopfield_batch(self.memories, x, k=k, beta=beta, hopfield_steps=hopfield_steps)



    def get_dists(self, Xi, x):
        # Xi in N x n
        # x in b x n

        # Duplicate Xi b times
        Xi = Xi.unsqueeze(0).repeat(x.shape[0], 1, 1)

        # add third dimension to x
        x = x.unsqueeze(1)

        # add batch dimension to X

        dists = torch.cdist(Xi, x, p=self.p)
        return dists.squeeze(2)

    @staticmethod
    def ssm(x, k, beta):
        # x in b x n
        return LML(N=k, n_iter = 200,eps = 1e-3)(beta * x)

    def k_softmax(self, beta, x, k):
        # x is b x N
        # beta is scalar
        # k is int
        # returns b x N x k
        assert(k>=1)
        result = torch.zeros(x.shape[0], x.shape[1], k).to(x.device)
        result[:, :, 0] =self.ssm(x, 1, beta)
        last_ssm = result[:, :, 0]
        for i in range(1, k):
            new_ssm  = self.ssm(x, i+1, beta)
            result[:, :, i] = new_ssm - last_ssm
            last_ssm = new_ssm.clone()
        return result

    def hopfield(self, Xi,  x, beta):
        # Xi is N x n
        # x is b x n

        return  torch.softmax(-1*beta * self.get_dists(Xi, x), dim=1) @ Xi

    
    def k_hopfield_batch(self, Xi, x, k, beta, hopfield_steps=0):
        # Xi is N x n
        # x is b x n
        # beta, k are scalar
        assert(Xi.shape[1] == x.shape[1])
        dists = self.get_dists(Xi, x)
        # dists is b x N
        ksm = self.k_softmax(beta, -1 *dists, k)
        # ksm is b x N x k
        # multiply Xi.t() by all b slices of ksm
        # result is b x n x k
        result = Xi.t() @ ksm

        for i in range(hopfield_steps):
            for j in range(k):
                result[:, :, j] = self.hopfield(Xi, result[:, :, j], beta)
        return result
    
    
    def soft_index(self, Xi, x, beta):
        # Xi is N x n
        # x is b x n
        # result is b x N
        print("beta:", beta)
        return  torch.softmax(-1*beta * self.get_dists(Xi, x), dim=1) 
    

    
    def k_soft_hopfield_index(self, Xi, x, k, beta, hopfield_steps=0):
                # Xi is N x n
        # x is b x n
        # beta, k are scalar
        assert(Xi.shape[1] == x.shape[1])

        dists = self.get_dists(Xi, x)
        # dists is b x N
        ksm = self.k_softmax(beta, -1 *dists, k)
        # ksm is b x N x k
        # multiply Xi.t() by all b slices of ksm
        # result is b x n x k
        if hopfield_steps == 0:
            return ksm
        else: 
            result = Xi.t() @ ksm

            for i in range(hopfield_steps):
                for j in range(k):
                    result[:, :, j] = self.soft_index(Xi, result[:, :, j], beta)
            return result
    