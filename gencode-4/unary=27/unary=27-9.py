import torch

class ClampingModel(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp to min value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp to max value
        return t3

# Initializing the model
model = ClampingModel(min_value=0.1, max_value=0.9)

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor
output_tensor = model(input_tensor)  # Forward pass

print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)
