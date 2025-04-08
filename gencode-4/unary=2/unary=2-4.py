import torch

# Define the model
class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5  # Multiply the output of the transposed convolution by 0.5
        t3 = t1 * t1 * t1 * 0.044715  # Multiply the output of the transposed convolution by itself twice, then by 0.044715
        t4 = t1 + t3  # Add the output of the transposed convolution to the result of the previous operation
        t5 = t4 * 0.7978845608028654  # Multiply the result of the previous operation by 0.7978845608028654
        t6 = torch.tanh(t5)  # Apply the hyperbolic tangent function to the result of the previous operation
        t7 = t6 + 1  # Add 1 to the result of the hyperbolic tangent function
        t8 = t2 * t7  # Multiply the output of the transposed convolution by the result of the previous operation
        return t8

# Initializing the model
model = ModelTransposeConv()

# Generating input tensor
# The input tensor should have the same number of channels as the output channels of the transposed convolution layer
# In this case, the output channels are 3, so we will use 8 channels for input
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Forward pass through the model
output = model(input_tensor)

# Output shape
print(output.shape)  # Should be [1, 3, 64, 64]
