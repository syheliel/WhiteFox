import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution; input channels = 8, output channels = 3, kernel size = 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution to the input tensor
        t2 = torch.sigmoid(t1)         # Apply the sigmoid function to the output of the transposed convolution
        t3 = t1 * t2                   # Multiply the output of the transposed convolution by the output of the sigmoid function
        return t3

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)  # Example input tensor with batch size 1, 8 channels, height 64, width 64
output = model(x1)

# Display the output shape
print(output.shape)  # Should be (1, 3, 64, 64) as we expect 3 output channels from the transposed convolution
