import torch

class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Apply hyperbolic tangent activation function
        t2 = torch.tanh(t1)
        return t2

# Initialize the model
model = TanhModel()

# Generating a random input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Get the output from the model
output_tensor = model(input_tensor)

# Print the output tensor shape
print(output_tensor.shape)  # Expected output shape: (1, 8, 64, 64)
