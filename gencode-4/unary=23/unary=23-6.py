import torch

# Model Definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # Apply pointwise transposed convolution
        t1 = self.conv_transpose(x)
        # Apply hyperbolic tangent function
        t2 = torch.tanh(t1)
        return t2

# Initializing the model
model = TransposeConvModel()

# Input tensor for the model
# The input tensor must have a channel size corresponding to the output channels of the previous layer
# For this example, we are assuming the input tensor has 8 channels (as defined by the ConvTranspose2d layer)
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the output shape
print("Output shape:", output_tensor.shape)
