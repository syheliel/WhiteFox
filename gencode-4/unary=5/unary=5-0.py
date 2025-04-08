import torch

# Define the model class
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)
        t2 = t1 * 0.5
        t3 = t1 * 0.7071067811865476
        t4 = torch.erf(t3)
        t5 = t4 + 1
        t6 = t2 * t5
        return t6

# Initializing the model
model = TransposeConvModel()

# Input tensor with batch size of 1, 3 input channels, and a spatial size of 64x64
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output = model(input_tensor)

# Print the shape of the output tensor
print(output.shape)
