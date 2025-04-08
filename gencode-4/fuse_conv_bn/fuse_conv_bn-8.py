import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        
        # Set the running mean and variance as constant attributes
        self.bn.running_mean = nn.Parameter(torch.zeros(8), requires_grad=False)
        self.bn.running_var = nn.Parameter(torch.ones(8), requires_grad=False)
        self.bn.weight = nn.Parameter(torch.ones(8), requires_grad=False)
        self.bn.bias = nn.Parameter(torch.zeros(8), requires_grad=False)

    def forward(self, x):
        # Set batch norm to evaluation mode
        self.bn.eval()
        conv_out = self.conv(x)
        output = self.bn(conv_out)
        return output

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor with shape (batch_size, channels, height, width)

# Getting the output from the model
output = model(input_tensor)
