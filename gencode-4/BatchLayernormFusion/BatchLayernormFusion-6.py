import torch

class NewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using a different number of input/output channels and kernel size
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Pointwise convolution with kernel size 1

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1
        t6 = t2 * t5  # Multiply the results
        return t6

# Initializing the model
new_model = NewModel()

# Generating input tensor
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 size

# Forward pass through the model
output = new_model(input_tensor)

# Print the output shape
print(output.shape)
