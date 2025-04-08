import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)

        # Initializing running mean and variance for demonstration purposes
        self.bn.running_mean = torch.zeros(8)
        self.bn.running_var = torch.ones(8)
        self.bn.weight = torch.ones(8)
        self.bn.bias = torch.zeros(8)

    def forward(self, x):
        self.bn.eval()  # Set BatchNorm to evaluation mode
        conv_out = self.conv(x)
        output = self.bn(conv_out)  # Apply batch normalization
        return output

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output_tensor = model(input_tensor)

print(output_tensor.shape)  # Output shape (1, 8, 64, 64)
