import torch

# Model Definition
class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * t1 * t1 * 0.044715  # Multiply by itself twice and by 0.044715
        t4 = t1 + t3  # Add the output of the transposed convolution to the result of the previous operation
        t5 = t4 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t6 = torch.tanh(t5)  # Apply the hyperbolic tangent function
        t7 = t6 + 1  # Add 1 to the result of the hyperbolic tangent function
        t8 = t2 * t7  # Multiply the output of the transposed convolution by the result of the previous operation
        return t8

# Initializing the model
model = ModelTransposeConv()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, height and width of 64
output = model(input_tensor)

# Display the output shape
print(output.shape)
