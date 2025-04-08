import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # 2D Convolution
        self.bn = torch.nn.BatchNorm2d(8)  # Batch Normalization for 2D

        # Set constant attributes for batch normalization
        self.bn.running_mean = torch.zeros(8)
        self.bn.running_var = torch.ones(8)
        self.bn.weight = torch.ones(8)
        self.bn.bias = torch.zeros(8)

    def forward(self, x):
        self.conv.eval()  # Set convolution layer to evaluation mode
        conv_output = self.conv(x)
        bn_output = self.bn(conv_output)  # Apply batch normalization
        return bn_output

# Initializing the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image

# Get the output from the model
output = model(input_tensor)
