import torch

class NewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Multiply by 0.5
        t2 = t1 * 0.5
        # Multiply by 0.7071067811865476
        t3 = t1 * 0.7071067811865476
        # Apply the error function
        t4 = torch.erf(t3)
        # Add 1
        t5 = t4 + 1
        # Multiply the two results
        t6 = t2 * t5
        return t6

# Initializing the model
new_model = NewModel()

# Input tensor to the model
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
output = new_model(input_tensor)

# Print the output shape
print(output.shape)
