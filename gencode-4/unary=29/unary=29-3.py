import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp to minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp to maximum value
        return t3

# Initializing the model
model = TransposeConvModel(min_value=0.0, max_value=1.0)

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass
output_tensor = model(input_tensor)

# Displaying the output tensor shape
print(output_tensor.shape)
