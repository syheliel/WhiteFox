import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Defining a 2D convolutional layer
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Defining a batch normalization layer
        self.bn = torch.nn.BatchNorm2d(num_features=8)
        
        # Setting the model to evaluation mode for BatchNorm
        self.bn.eval()
        
        # Dummy running mean and variance for batch normalization
        self.bn.running_mean = torch.zeros(8)
        self.bn.running_var = torch.ones(8)
        self.bn.weight = torch.ones(8)
        self.bn.bias = torch.zeros(8)

    def forward(self, x):
        # Apply convolution and batch normalization
        x = self.conv(x)
        x = self.bn(x)
        return x

# Initializing the model
model = Model()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Getting the output from the model
output = model(input_tensor)

# Print output shape
print(output.shape)
