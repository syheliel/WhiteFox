import torch

# Model definition
class SimpleTanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.tanh(t1)  # Apply hyperbolic tangent activation
        return t2

# Initializing the model
model = SimpleTanhModel()

# Generating an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Passing the input tensor through the model
output_tensor = model(input_tensor)

# Displaying the output tensor shape
print("Output tensor shape:", output_tensor.shape)
