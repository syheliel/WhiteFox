import torch

# Model Definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using a pointwise transposed convolution
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * t1 * t1 * 0.044715  # Multiply t1 by itself twice and by 0.044715
        t4 = t1 + t3  # Add t1 and t3
        t5 = t4 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t6 = torch.tanh(t5)  # Apply the hyperbolic tangent function
        t7 = t6 + 1  # Add 1 to the result of tanh
        t8 = t2 * t7  # Multiply t2 by t7
        return t8

# Initializing the model
model = TransposeConvModel()

# Generate Input Tensor
# The input tensor should have the shape (batch_size, channels, height, width)
# In this case, we assume a batch size of 1, 8 input channels, and a spatial size of 64x64
input_tensor = torch.randn(1, 8, 64, 64)

# Get the output of the model
output = model(input_tensor)

# Print the output shape
print("Output shape:", output.shape)
