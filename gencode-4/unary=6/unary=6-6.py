import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)  # Pointwise convolution with kernel size 1

    def forward(self, x1):
        t1 = self.conv(x1)            # Apply pointwise convolution
        t2 = t1 + 3                   # Add 3 to the output of the convolution
        t3 = torch.clamp_min(t2, 0)   # Clamp to min 0
        t4 = torch.clamp_max(t3, 6)   # Clamp to max 6
        t5 = t1 * t4                  # Multiply by the output of the clamp operation
        t6 = t5 / 6                   # Divide by 6
        return t6                     # Return the final output

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Random input tensor with shape [batch_size, channels, height, width]

# Forward pass
output = model(input_tensor)

# Print the output shape to verify
print("Output shape:", output.shape)
