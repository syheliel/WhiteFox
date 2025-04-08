import torch

class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * t1 * t1 * 0.044715  # Multiply by itself twice and by 0.044715
        t4 = t1 + t3  # Add the previous results
        t5 = t4 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t6 = torch.tanh(t5)  # Apply tanh
        t7 = t6 + 1  # Add 1
        t8 = t2 * t7  # Multiply the output of the transposed convolution by the result of the previous operation
        return t8

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
# Generate a random input tensor with shape (1, 8, 64, 64)
x1 = torch.randn(1, 8, 64, 64)
output = model(x1)

# Output shape
print("Output shape:", output.shape)
