import torch

class NewModel(torch.nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        v1 = self.conv(x)  # Apply pointwise convolution
        v2 = v1 * 0.5     # Multiply by 0.5
        v3 = v1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        v4 = torch.erf(v3)  # Apply the error function
        v5 = v4 + 1         # Add 1
        v6 = v2 * v5        # Final multiplication
        return v6

# Initializing the model
model = NewModel()

# Input to the model (batch_size=1, channels=3, height=128, width=128)
input_tensor = torch.randn(1, 3, 128, 128)

# Forward pass through the model
output = model(input_tensor)

# Print the shape of the output tensor
print(output.shape)
