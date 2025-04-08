import torch
import torch.nn.functional as F

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)

        # Initialize batch normalization parameters
        self.bn.running_mean = torch.zeros(16)
        self.bn.running_var = torch.ones(16)
        self.bn.weight = torch.ones(16)
        self.bn.bias = torch.zeros(16)

        # Set batch normalization to evaluation mode
        self.bn.eval()

    def forward(self, x):
        conv_output = self.conv(x)
        output = self.bn(conv_output)
        return output

# Initializing the model
model = Model()

# Generating an input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output = model(input_tensor)

# Output shape for verification
print("Output shape:", output.shape)
