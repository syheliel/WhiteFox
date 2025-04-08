import torch

# Define the model
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 3, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)          # Apply pointwise transposed convolution
        t2 = torch.tanh(t1)                  # Apply the hyperbolic tangent function
        return t2

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
input_tensor = torch.randn(1, 16, 64, 64)  # Batch size of 1, 16 channels, 64x64 spatial dimensions
output_tensor = model(input_tensor)

print("Output shape:", output_tensor.shape)
