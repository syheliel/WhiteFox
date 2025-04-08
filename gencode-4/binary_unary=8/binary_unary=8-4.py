import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)  # Pointwise convolution (1x1)

    def forward(self, x1, other):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = t1 + other     # Add another tensor to the output of the convolution
        t3 = torch.nn.functional.relu(t2)  # Apply ReLU activation function
        return t3

# Initializing the model
model = Model()

# Generating input tensor
x1 = torch.randn(1, 3, 64, 64)  # Input tensor with batch size 1, 3 channels, 64x64 resolution
other = torch.randn(1, 16, 64, 64)  # Another tensor to be added, matching the output channels of the convolution

# Forward pass
output = model(x1, other)

# Output shape
print(output.shape)
