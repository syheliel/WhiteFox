import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(num_features=8)  # BatchNorm2d matches the Conv2d

        # Initialize running mean and variance, weight and bias with constant attributes
        self.bn.running_mean = torch.zeros(8)  # Assuming 8 channels
        self.bn.running_var = torch.ones(8)    # Assuming 8 channels
        self.bn.weight = torch.ones(8)         # Assuming 8 channels
        self.bn.bias = torch.zeros(8)          # Assuming 8 channels

    def forward(self, x):
        self.conv.eval()  # Set convolution in evaluation mode
        conv_output = self.conv(x)
        bn_output = self.bn(conv_output)  # Apply batch normalization
        return bn_output

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

print(output.shape)  # Output shape
