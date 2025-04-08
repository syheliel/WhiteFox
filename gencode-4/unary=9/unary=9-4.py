import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)                # Apply pointwise convolution
        t2 = t1 + 3                      # Add 3 to the output
        t3 = torch.clamp_min(t2, 0)     # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)     # Clamp the output to a maximum of 6
        t5 = t4 / 6                      # Divide the output by 6
        return t5

# Initializing the model
model = Model()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Getting the output from the model
output = model(input_tensor)

# Output the result
print(output)
