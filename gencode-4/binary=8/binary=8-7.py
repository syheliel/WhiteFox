import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x1, other):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = t1 + other     # Add another tensor to the output of the convolution
        return t2

# Initializing the model
model = Model()

# Generate input tensor for the model
x1 = torch.randn(1, 3, 64, 64)  # Input tensor with shape (batch_size, channels, height, width)

# Generate the 'other' tensor to be added (should have the same shape as the output of the convolution)
other = torch.randn(1, 8, 64, 64)  # Output shape from conv layer will be (1, 8, 64, 64)

# Forward pass
output = model(x1, other)
