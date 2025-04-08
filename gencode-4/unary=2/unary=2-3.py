import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5  # Multiply the output of the transposed convolution by 0.5
        t3 = t1 * t1 * t1 * 0.044715  # Multiply the output of the transposed convolution by itself twice and then by 0.044715
        t4 = t1 + t3  # Add the output of the transposed convolution to the result of the previous operation
        t5 = t4 * 0.7978845608028654  # Multiply the result of the previous operation by 0.7978845608028654
        t6 = torch.tanh(t5)  # Apply the hyperbolic tangent function to the result of the previous operation
        t7 = t6 + 1  # Add 1 to the result of the hyperbolic tangent function
        t8 = t2 * t7  # Multiply the output of the transposed convolution by the result of the previous operation
        return t8

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
input_tensor = torch.randn(1, 8, 64, 64)  # A batch of 1 with 8 channels and 64x64 spatial dimensions
output = model(input_tensor)

# Verifying output shape
print("Output shape:", output.shape)
