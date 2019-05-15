'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/time_distributed.py
Given an input shaped like (batch_size, sequence_length, ...) and a Module that takes input like (batch_size, ...)
TimeDistributed can reshape the input to be (batch_size * sequence_length, ...) applies the Module, then reshape back.
Update date: 2019-03-02'''
import torch

class TimeDistributed(torch.nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self._module = module

    def forward(self, *inputs):
        reshaped_inputs = []

        for input_tensor in inputs:
            input_size = input_tensor.size()
            assert len(input_size) > 2
            squashed_shape = [-1] + [x for x in input_size[2:]]
            reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))

        reshaped_outputs = self._module(*reshaped_inputs)

        original_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
        outputs = reshaped_outputs.contiguous().view(*original_shape)
        return outputs