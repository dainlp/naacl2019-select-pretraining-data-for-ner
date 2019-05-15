import itertools
import torch
from typing import List

def block_orthogonal(tensor, split_sizes: List[int], gain=1.0):
    # tensor: the tensor to initialize
    # split_sizes: [10, 20] result in the tensor being split into chunks of size 10 along the first dimension
    # 20 along the second
    # Used in the case of recurrent models which use multiple gates applied to linear projections.
    # Separate parameters should be initialized independently
    data = tensor.data
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("block_orthogonal: tensor size and split sizes not compatible!")

    indexes = [list(range(0, max_size, split)) for max_size, split in zip(sizes, split_sizes)]
    for block_state_indices in itertools.product(*indexes):
        index_and_step_tuples = zip(block_state_indices, split_sizes)
        block_slice = tuple([slice(start_index, start_index + step) for start_index, step in index_and_step_tuples])
        data[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)
