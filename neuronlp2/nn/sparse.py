__author__ = 'Dung Doan'

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

def assign_tensor(tensor, val):

    if isinstance(tensor, Variable):
        assign_tensor(tensor.data, val)
        return tensor
    return tensor.copy_(val)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_embedding=None, freeze=False, padding_idx=None,
               max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.frozen = freeze
        self.sparse = sparse

        self.reset_parameters(init_embedding)

    def reset_parameters(self, init_embedding):
        if init_embedding is None:
            scale = np.sqrt(3.0 / self.embeddings_dim)
            self.weight.data.uniform_(-scale, scale)
        else:
            assign_tensor(self.weight, init_embedding)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

        if self.frozen:
            if init_embedding is None:
                raise Warning('Freeze embeddings which are randomly initialized')
            self.weight.requires_grad = False

    def freeze(self):
        self.weight.requires_grad = False
        self.frozen = True

    def forward(self, input):
        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1

        input_size = input.size()
        if input.dim() > 2:
            num_inputs = int(np.prod(input_size[:-1]))
            input = input.view(num_inputs, input_size[-1])

        output_size = input_size + (self.embeddings_dim,)
        return self._backend.Embedding.apply(
            input, self.weight, padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).view(output_size)

    def __repr__(self):
        s = '{name}{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

