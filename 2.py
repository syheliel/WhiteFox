import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch as th
import torch.linalg as la
from torch.nn import Parameter
import torch.linalg as linalg

class VulnerableModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)

    def forward(self, x):
        graph = torch.fx.Graph()
        input_node = graph.placeholder('x')
        conv_node = graph.call_module('conv', (input_node,))
        invalid_node = graph.create_node(op='call_module', target=123, args=(conv_node,))
        try:
            target = get_node_target(self._modules, invalid_node)
        except Exception as e:
            print(f'Error occurred: {e}')
        return conv_node


func = VulnerableModel().to('cuda:2')


input_tensor = torch.randn(1, 3, 32, 32)

test_inputs = [input_tensor]

func(input_tensor)

# JIT_FAIL
'''
direct:


jit:

'''