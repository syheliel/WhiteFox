import torch

# Model
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 + 3                   # Add 3 to the output of the transposed convolution
        t3 = torch.clamp_min(t2, 0)   # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)    # Clamp the output to a maximum of 6
        t5 = t4 / 6                     # Divide the output by 6
        return t5

# Initializing the model
model = TransposedConvModel()

# Inputs to the model
# Random input tensor with shape [1, 8, 64, 64]
input_tensor = torch.randn(1, 8, 64, 64)

# Forward pass to get the output
output = model(input_tensor)

# Print output shape
print(output.shape)
