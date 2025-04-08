import torch

class TransposedConvModel(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__()
        self.trans_conv = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        t1 = self.trans_conv(x)  # Apply pointwise transposed convolution
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp to minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp to maximum value
        return t3

# Initializing the model
model = TransposedConvModel(min_value=0.0, max_value=1.0)

# Input tensor to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Pass the input tensor through the model
output_tensor = model(input_tensor)

# Output the result
print(output_tensor.shape)  # Output shape should be [1, 8, 64, 64]
