import torch

class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution: input channels=8, output channels=3, kernel size=1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x1):
        t1 = self.conv_transpose(x1)           # Apply pointwise transposed convolution
        t2 = t1 + 3                             # Add 3 to the output
        t3 = torch.clamp_min(t2, 0)            # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)            # Clamp the output to a maximum of 6
        t5 = t4 / 6                              # Divide the output by 6
        return t5

# Initializing the model
model = TransposedConvModel()

# Inputs to the model: 1 sample, 8 channels, 64x64 spatial dimensions
input_tensor = torch.randn(1, 8, 64, 64)

# Get the output from the model
output_tensor = model(input_tensor)

print(output_tensor.shape)  # Should print the shape of the output tensor

input_tensor = torch.randn(1, 8, 64, 64)  # 1 sample, 8 channels, 64x64 spatial dimensions

print(output_tensor.shape)  # This will provide the shape of the output tensor after processing through the model.
