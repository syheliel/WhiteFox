import torch
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)  # 2D Convolution
        self.bn = torch.nn.BatchNorm2d(num_features=8)  # Batch Normalization for 2D

        # Initialize running mean, running variance, weight, and bias as constant attributes
        self.bn.running_mean = torch.zeros(8)
        self.bn.running_var = torch.ones(8)
        self.bn.weight = torch.ones(8)
        self.bn.bias = torch.zeros(8)

    def forward(self, x):
        # Set the model in evaluation mode
        self.eval()
        conv_output = self.conv(x)  # Apply convolution
        output = self.bn(conv_output)  # Apply batch normalization
        return output

# Initializing the model
model = MyModel()

# Input tensor for the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image size

# Getting the output from the model
output = model(input_tensor)
print(output.shape)  # To verify the output shape
