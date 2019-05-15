import torch
import torch.nn as nn

'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/highway.py
https://github.com/LiyuanLucasLiu/LM-LSTM-CRF/blob/master/model/highway.py
A gated combination of a linear transformation and a non-linear transformation of its input.
Math: y = g * x + (1 - g) * f(A(x)),
g is an element-wise gate, computed as: sigmoid(B(x)).
A is a linear transformation, f is an element-wise non-linearity
Update date: April-19-2019'''
class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=1):
        super(Highway, self).__init__()

        self._layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        for layer in self._layers:
            # Bias the highway layer to just carry its input forward.
            # Set the bias on B(x) to be positive, then g will be biased to be high
            # The bias on B(x) is the second half of the bias vector in each linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        current_inputs = inputs
        for layer in self._layers:
            linear_part = current_inputs
            projected_inputs = layer(current_inputs)

            nonlinear_part, gate = projected_inputs.chunk(2, dim=-1)
            nonlinear_part = nn.functional.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_inputs = gate * linear_part + (1 - gate) * nonlinear_part
        return current_inputs


if __name__ == "__main__":
    highway = Highway(input_dim=5, num_layers=2)
    input = torch.rand(3, 5)
    print(input)
    output = highway(input)
    print(output)