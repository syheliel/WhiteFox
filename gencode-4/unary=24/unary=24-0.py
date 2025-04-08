import torch

# Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)                      # Apply pointwise convolution
        t2 = t1 > 0                            # Create a boolean mask
        negative_slope = 0.01                  # Define a negative slope for Leaky ReLU
        t3 = t1 * negative_slope               # Multiply by negative slope
        t4 = torch.where(t2, t1, t3)          # Apply the where function
        return t4

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

# Print output shape to verify
print("Output shape:", output.shape)
