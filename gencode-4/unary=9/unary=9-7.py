import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 + 3                   # Add 3 to the output of the convolution
        t3 = torch.clamp_min(t2, 0)  # Clamp the output of the addition to a minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp the output to a maximum of 6
        t5 = t4 / 6                   # Divide the output by 6
        return t5

# Initializing the model
model = Model()

# Generating an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 spatial dimensions

# Forward pass through the model
output = model(input_tensor)

# Output the shape of the output tensor
print(output.shape)
