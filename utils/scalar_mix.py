'''
Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py
Compute a parameterised scalar mixture of N tensors:
        outs = gamma * sum(s_k * tensor_k)
        s_k = softmax(w)
        gamma and w are parameters
        Imagine tensor_k are outputs of each layer in ELMo, and outs is its final weighted (s_k) representation.
Update date: 2019-02-27
'''

import torch
from torch.nn import ParameterList, Parameter

class ScalarMix(torch.nn.Module):
    def __init__(self, num_tensors, trainable=True):
        super(ScalarMix, self).__init__()
        self.num_tensors = num_tensors
        self.scalar_parameters = ParameterList([Parameter(torch.FloatTensor([0.0]), requires_grad=trainable) for _ in range(num_tensors)])
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(self, tensors):
        # tensors must all be the same shape, let's say (batch_size, timesteps, dim)
        assert self.num_tensors == len(tensors)

        normed_weights = torch.nn.functional.softmax(torch.cat([p for p in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)
