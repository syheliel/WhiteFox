import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        
        # Setting constants for batch normalization running stats
        self.bn.running_mean = torch.zeros(8)  # Assuming batch size of 8 channels
        self.bn.running_var = torch.ones(8)    # Variance of 1
        self.bn.weight = torch.ones(8)         # Weight of 1
        self.bn.bias = torch.zeros(8)          # Bias of 0

    def forward(self, x):
        self.conv.eval()  # Set convolution to evaluation mode
        # Apply convolution followed by batch normalization
        conv_output = self.conv(x)
        output = self.bn(conv_output)
        return output

# Initializing the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output = model(input_tensor)

# Output the shape of the output tensor
print(output.shape)
