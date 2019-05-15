'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/attention/*
Update date: 2019-04-01'''

import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .pytorch import masked_softmax


__all__ = ["BilinearAttention"]


class Attention(torch.nn.Module):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self._normalize = normalize

    def forward(self, vector, matrix, matrix_mask=None):
        similarities = self._forward_internal(vector, matrix)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities

    def _forward_internal(self, vector, matrix):
        raise NotImplementedError


class BilinearAttention(Attention):
    '''The similarity between the vector x and the matrix y is: x^T W y + b, where W, b are parameters'''
    def __init__(self, vector_dim, matrix_dim):
        super().__init__()
        self._W = Parameter(torch.Tensor(vector_dim, matrix_dim))
        self._b = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._W)
        self._b.data.fill_(0)

    def _forward_internal(self, vector, matrix):
        intermediate = vector.mm(self._W).unsqueeze(1)
        return intermediate.bmm(matrix.transpose(1, 2)).squeeze(1) + self._b


'''Reference url: http://nlp.seas.harvard.edu/2018/04/03/attention.html
Attention Is All You Need Section 3.2.2
Update date: 2019-04-06'''
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h

        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = ScaledDotProduct.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)


'''Reference url: http://nlp.seas.harvard.edu/2018/04/03/attention.html
Attention Is All You Need Section 3.2.1 
Update date: 2019-04-06'''
class ScaledDotProduct(object):
    @classmethod
    def attention(cls, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


if __name__ == "__main__":
    vector_dim = 5
    matrix_dim = 10
    batch_size = 2
    sequence_length = 8
    attention = BilinearAttention(5, 10)
    vector = torch.rand(batch_size, vector_dim)
    matrix = torch.rand(batch_size, sequence_length, matrix_dim)
    attention_weights = attention(vector, matrix)
    print(attention_weights)
    print(attention_weights.shape)