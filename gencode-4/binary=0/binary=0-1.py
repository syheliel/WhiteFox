import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x1, other):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = t1 + other  # Add another tensor to the output of the convolution
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)  # Input tensor with batch size 1, 3 channels, 64x64 dimensions
other = torch.randn(1, 8, 64, 64)  # Other tensor to be added, must match the output shape of the convolution

# Get output from the model
output = m(x1, other)

print(output.shape)  # This will show the shape of the output tensor
