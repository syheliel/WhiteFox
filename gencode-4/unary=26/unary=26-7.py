import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)
        self.negative_slope = 0.01  # Define the negative slope for the Leaky ReLU

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 > 0  # Create a mask where the output of the convolution is greater than 0
        t3 = t1 * self.negative_slope  # Multiply the output of the convolution by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the mask to the output of the convolution
        return t4

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions
output_tensor = model(input_tensor)

# Display output shape
print("Output shape:", output_tensor.shape)
