import torch

# Model Definition
class GatedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)          # Apply pointwise convolution
        t2 = torch.sigmoid(t1)    # Apply sigmoid to the convolution output
        t3 = t1 * t2               # Multiply the output of the convolution by the sigmoid output
        return t3

# Initializing the model
model = GatedConvModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # A random input tensor with shape (N, C, H, W)
output = model(input_tensor)               # Forward pass through the model

# Output shape
print("Output shape:", output.shape)
