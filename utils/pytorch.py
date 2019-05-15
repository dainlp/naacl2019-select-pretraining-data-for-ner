from collections import defaultdict
import copy
import torch
import torch.nn as nn
from typing import Dict, List


__all__ = ["batch_tensor_dicts", "clones",
           "get_device_of", "get_lengths_from_binary_sequence_mask",
           "has_tensor", "move_to_gpu"]


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-April-26'''
def batch_tensor_dicts(tensor_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    key_to_tensors = defaultdict(list)
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            key_to_tensors[key].append(tensor)

    batched_tensors = {}
    for key, tensor_list in key_to_tensors.items():
        batched_tensors[key] = torch.stack(tensor_list)

    return batched_tensors


'''Reference url: http://nlp.seas.harvard.edu/2018/04/03/attention.html
Update date: 2019-April-26'''
def clones(module: torch.nn.Module, N: int) -> torch.nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-April-26'''
def get_device_of(tensor) -> int:
    if not tensor.is_cuda:
        return -1
    return tensor.get_device()


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
inputs:
    mask: binary mask of shape (batch_size, max_sequence_length)
outputs:
    torch.LongTensor of shape (batch_size, )
Update date: 2019-April-26'''
def get_lengths_from_binary_sequence_mask(mask: torch.Tensor):
    return mask.long().sum(-1)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-April-26'''
def has_tensor(obj) -> bool:
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, dict):
        return any(has_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(has_tensor(i) for i in obj)
    return False


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-April-26'''
def masked_softmax(vector, mask=None):
    if mask is None:
        return torch.nn.functional.softmax(vector, dim=-1)
    else:
        mask = mask.float()
        assert mask.dim() == vector.dim()
        # use a very large negative number for those masked positions
        # so that the probabilities of those positions would be approximately 0.
        # This is not accurate in math, but works for most cases and consumes less memory.
        masked_vector = vector.masked_fill((1 - mask).byte(), -1e32)
        return torch.nn.functional.softmax(masked_vector, dim=-1)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/metric.py
Update date: 2019-03-01'''
def move_to_cpu(*tensors):
    return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py (move_to_device)
Update date: 2019-April-26'''
def move_to_gpu(obj, cuda_device=0):
    if cuda_device < 0 or not has_tensor(obj): return obj
    if isinstance(obj, torch.Tensor): return obj.cuda(cuda_device)
    if isinstance(obj, dict):
        return {k: move_to_gpu(v, cuda_device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_gpu(v, cuda_device) for v in obj]
    if isinstance(obj, tuple):
        return tuple([move_to_gpu(v, cuda_device) for v in obj])
    return obj


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/util.py
Update date: 2019-03-03'''
def rescale_gradients(model, grad_norm=5.0):
    if grad_norm:
        parameters = [p for p in model.parameters() if p.grad is not None]
        return sparse_clip_norm(parameters, grad_norm)
    return None


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/util.py
Update date: 2019-03-03'''
def sparse_clip_norm(parameters, max_norm: float, norm_type=2.0):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0
    for p in parameters:
        if p.grad.is_sparse:
            grad = p.grad.data.coalesce()
            param_norm = grad._values().norm(norm_type)
        else:
            param_norm = p.grad.data.norm(norm_type)

        total_norm += param_norm ** norm_type

    total_norm = total_norm ** (1. / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad.is_sparse:
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)

    return total_norm