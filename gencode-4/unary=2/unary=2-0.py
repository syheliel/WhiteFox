import torch

# Define the model
class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * t1 * t1 * 0.044715  # Multiply by itself twice and then by 0.044715
        t4 = t1 + t3  # Add the two results
        t5 = t4 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t6 = torch.tanh(t5)  # Apply the hyperbolic tangent function
        t7 = t6 + 1  # Add 1
        t8 = t2 * t7  # Multiply the output of the transposed convolution by the result
        return t8

# Initializing the model
model = ModelTransposeConv()

# Generate input tensor for the model
# The input tensor should have the shape (batch_size, channels, height, width)
# Here, we assume a batch size of 1, 8 input channels, and a spatial size of 64x64
input_tensor = torch.randn(1, 8, 64, 64)

# Forward pass through the model
output_tensor = model(input_tensor)

# Display output tensor shape
print(output_tensor.shape)
