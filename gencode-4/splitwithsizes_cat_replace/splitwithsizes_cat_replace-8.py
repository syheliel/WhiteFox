import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution layer with weights and optional bias
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Predefined scalar tensor (attribute)
        self.scalar = torch.tensor(2.0, dtype=torch.float32)

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply convolution
        t2 = t1 * self.scalar          # Multiply with predefined scalar
        return t2

# Initialize the model
model = CustomModel()

# Generate an input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Get the output from the model
output = model(input_tensor)
