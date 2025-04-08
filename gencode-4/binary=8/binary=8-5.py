import torch

# Define the Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x1, other):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = t1 + other     # Add the other tensor to the output of the convolution
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)  # Input tensor with shape (batch_size, channels, height, width)
other = torch.randn(1, 8, 64, 64)  # Another tensor to add, with the same shape as the convolution output

# Forward pass
__output__ = m(x1, other)

# Print output shape
print(__output__.shape)
