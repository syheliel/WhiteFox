import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
 
    def forward(self, input_tensor, other_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution with kernel size 1
        t2 = t1 + other_tensor         # Add another tensor to the output of the convolution
        return t2

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)          # Input tensor with batch size of 1, 3 channels, and 64x64 spatial dimensions
other_tensor = torch.randn(1, 8, 64, 64)           # Another tensor with the same spatial dimensions and output channels as the convolution output

# Forward pass through the model
output = model(input_tensor, other_tensor)

# Output shape
print(output.shape)
