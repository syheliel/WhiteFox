import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a convolution layer with 3 input channels, 8 output channels, and a kernel size of 3
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        # Define an attribute for the other tensor to be added (scalar value)
        self.add_scalar = torch.tensor(1.5, dtype=torch.float32)

    def forward(self, x1):
        # Apply convolution to the input tensor
        t1 = self.conv(x1)
        # Apply addition operation with the scalar
        t2 = t1 + self.add_scalar
        return t2

# Initializing the model
m = Model()

# Generate an input tensor with shape (batch_size, channels, height, width)
x1 = torch.randn(1, 3, 64, 64)

# Get the output of the model
__output__ = m(x1)

# Print the output shape to verify
print(__output__.shape)
