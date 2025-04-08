import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x, min_value=0.0, max_value=1.0):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.clamp_min(t1, min_value)  # Clamp the output to a minimum value
        t3 = torch.clamp_max(t2, max_value)  # Clamp the output to a maximum value
        return t3

# Initialize the model
model = TransposeConvModel()

# Generate an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 spatial dimensions

# Pass the input tensor through the model
output_tensor = model(input_tensor, min_value=0.0, max_value=1.0)

# Display the shape of the output tensor
print("Output tensor shape:", output_tensor.shape)
