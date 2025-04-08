import torch

# Define the model
class ClampingModel(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp to minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp to maximum value
        return t3

# Initializing the model with specific min and max values
min_value = 0.0
max_value = 1.0
model = ClampingModel(min_value=min_value, max_value=max_value)

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

# Output the resulting tensor
print(output)
