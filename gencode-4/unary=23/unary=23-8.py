import torch

# Defining the model
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using a pointwise transposed convolution
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x):
        v1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        v2 = torch.tanh(v1)  # Apply the hyperbolic tangent function
        return v2

# Initializing the model
model = TransposedConvModel()

# Generating the input tensor
# Input tensor should have the number of channels matching the output channels of the previous layer.
# Here, we are assuming the input to be (batch_size, channels, height, width)
input_tensor = torch.randn(1, 8, 64, 64)  # Example input tensor

# Forward pass through the model
output = model(input_tensor)

# Print output shape for verification
print("Output shape:", output.shape)
