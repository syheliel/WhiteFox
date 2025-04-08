import torch

# Define the model class
class NewModel(torch.nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        # Applying the specified transformations
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1
        t6 = t2 * t5  # Multiply the two results
        return t6

# Initialize the model
model = NewModel()

# Generate input tensor: a batch of 1 image with 3 channels and size 64x64
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output = model(input_tensor)

# Display the output shape
print("Output shape:", output.shape)
