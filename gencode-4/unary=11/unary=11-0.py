import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 + 3                    # Add 3 to the output
        t3 = torch.clamp_min(t2, 0)    # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)    # Clamp the output to a maximum of 6
        t5 = t4 / 6                     # Divide by 6
        return t5

# Initializing the model
model = Model()

# Generating an input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output = model(input_tensor)

print("Output shape:", output.shape)
