import torch

# Model Definition
class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution with kernel size 1 to the input tensor
        t2 = torch.tanh(t1)  # Apply the hyperbolic tangent function to the output of the convolution
        return t2

# Initializing the model
model = TanhModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Random input tensor with shape (batch_size, channels, height, width)
output_tensor = model(input_tensor)  # Forward pass through the model

# Displaying the output shape
print("Output shape:", output_tensor.shape)
