import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 - 0.3  # Subtract scalar 'other'
        t3 = torch.relu(t2)  # Apply ReLU activation
        return t3

# Initializing the model
model = CustomModel()

# Input to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output_tensor = model(input_tensor)

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Output tensor shape: {output_tensor.shape}")
