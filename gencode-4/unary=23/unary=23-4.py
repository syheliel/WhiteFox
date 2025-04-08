import torch

# Define the model
class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution (kernel size 1, stride 1, padding 0)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.tanh(t1)           # Apply hyperbolic tangent function
        return t2

# Initializing the model
model = ModelTransposeConv()

# Sample input tensor
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Forward pass through the model
output = model(input_tensor)

# Print the shape of the output
print(output.shape)  # Should print: torch.Size([1, 3, 64, 64])
