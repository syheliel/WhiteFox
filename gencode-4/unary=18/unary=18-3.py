import torch

class SigmoidModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)          # Apply pointwise convolution
        t2 = torch.sigmoid(t1)     # Apply sigmoid activation
        return t2

# Initializing the model
model = SigmoidModel()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 color channels, 64x64 image size

# Forward pass
output = model(input_tensor)

# Output tensor shape
print("Output tensor shape:", output.shape)
