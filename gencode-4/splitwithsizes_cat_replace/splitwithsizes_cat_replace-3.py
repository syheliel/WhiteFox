import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Define a scalar attribute for the binary operation
        self.scalar = torch.tensor(2.5, dtype=torch.float32)

    def forward(self, x):
        # Apply convolution to the input tensor
        t1 = self.conv(x)
        # Apply binary operation (multiplication with scalar)
        t2 = t1 * self.scalar
        return t2

# Initializing the model
model = CustomModel()

# Input tensor to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)
