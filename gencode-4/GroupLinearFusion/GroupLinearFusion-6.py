import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a pointwise convolution layer
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

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
        # Multiply the results
        t6 = t2 * t5
        return t6

# Initializing the model
model = CustomModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Get the output from the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
