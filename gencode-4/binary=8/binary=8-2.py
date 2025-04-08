import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x1, other):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = t1 + other  # Add another tensor to the output of the convolution
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)  # Input tensor
other = torch.randn(1, 8, 64, 64)  # Another tensor of the same shape as the output of the convolution

# Model output
output = m(x1, other)
