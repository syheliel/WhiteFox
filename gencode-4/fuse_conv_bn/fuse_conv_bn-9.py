import torch
import torch.nn.functional as F

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 2D Convolution layer with 3 input channels and 16 output channels
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Batch normalization for 16 channels
        self.bn = torch.nn.BatchNorm2d(16)
        
        # Set the model to evaluation mode
        self.eval()

        # Manually set running mean and variance for batch normalization
        with torch.no_grad():
            self.bn.running_mean = torch.zeros(16)
            self.bn.running_var = torch.ones(16)
            self.bn.weight = torch.ones(16)
            self.bn.bias = torch.zeros(16)

    def forward(self, x):
        # Apply convolution followed by batch normalization
        x = self.conv(x)
        x = self.bn(x)
        return x

# Initializing the model
model = CustomModel()

# Generating input tensor with the shape (1, 3, 64, 64)
input_tensor = torch.randn(1, 3, 64, 64)

# Getting the output from the model
output = model(input_tensor)
