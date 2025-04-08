import torch

class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * t1 * t1 * 0.044715  # Multiply by itself twice and then by 0.044715
        t4 = t1 + t3  # Add the results
        t5 = t4 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t6 = torch.tanh(t5)  # Apply hyperbolic tangent
        t7 = t6 + 1  # Add 1
        t8 = t2 * t7  # Final multiplication
        return t8

# Initializing the model
model = ModelTransposeConv()

# Input to the model
x1 = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 size
output = model(x1)

# Output tensor shape
print(output.shape)
