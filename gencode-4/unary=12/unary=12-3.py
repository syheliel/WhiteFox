import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedModel(nn.Module):
    def __init__(self):
        super(GatedModel, self).__init__()
        # Pointwise convolution with kernel size 1
        self.conv = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.sigmoid(t1)  # Apply sigmoid function
        t3 = t1 * t2  # Multiply convolution output by sigmoid output
        return t3

# Initializing the model
model = GatedModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 input channels, 64x64 image
output_tensor = model(input_tensor)

# Output the result
print(output_tensor.shape)  # This should output: torch.Size([1, 8, 64, 64])

input_tensor = torch.randn(1, 3, 64, 64)  # A random tensor with the specified shape
