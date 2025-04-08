import torch

# Model definition
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Applying a pointwise transposed convolution
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise transposed convolution
        t1 = self.conv_transpose(x)
        # Apply the hyperbolic tangent function
        t2 = torch.tanh(t1)
        return t2

# Initializing the model
model = TransposedConvModel()

# Generating input tensor
# Let's assume the input tensor has 8 channels and a spatial dimension of 32x32
input_tensor = torch.randn(1, 8, 32, 32)

# Obtaining the output from the model
output_tensor = model(input_tensor)

# Displaying the shape of the output tensor
print("Output tensor shape:", output_tensor.shape)
