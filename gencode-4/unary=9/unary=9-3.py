import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = t1 + 3  # Add 3
        t3 = torch.clamp_min(t2, 0)  # Clamp to minimum 0
        t4 = torch.clamp_max(t3, 6)  # Clamp to maximum 6
        t5 = t4 / 6  # Divide by 6
        return t5

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
output = model(input_tensor)

print("Output shape:", output.shape)
