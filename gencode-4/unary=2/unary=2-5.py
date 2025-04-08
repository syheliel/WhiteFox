import torch

class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * t1 * t1 * 0.044715  # Multiply by itself twice and then by 0.044715
        t4 = t1 + t3  # Add the results of previous operation
        t5 = t4 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t6 = torch.tanh(t5)  # Apply hyperbolic tangent
        t7 = t6 + 1  # Add 1
        t8 = t2 * t7  # Multiply the output by the result
        return t8

# Initializing the model
model = TransposedConvModel()

# Inputs to the model
x_input = torch.randn(1, 3, 64, 64)  # A random input tensor with shape (batch_size, channels, height, width)
output = model(x_input)
