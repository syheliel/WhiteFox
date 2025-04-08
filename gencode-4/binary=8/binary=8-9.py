import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x1, other):
        # Apply pointwise convolution
        t1 = self.conv(x1)
        # Add another tensor to the output of the convolution
        t2 = t1 + other
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)           # Input tensor of shape (batch_size, channels, height, width)
other = torch.randn(1, 8, 64, 64)         # Another tensor to be added, shape must match the output of conv

# Forward pass
output = m(x1, other)

# Output shape
print(output.shape)
