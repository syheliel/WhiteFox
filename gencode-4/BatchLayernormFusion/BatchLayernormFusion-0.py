import torch

# Define the model class
class NewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Applying the specified operations
        t1 = self.conv(x)  # Pointwise convolution
        t2 = t1 * 0.5      # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3) # Apply the error function
        t5 = t4 + 1        # Add 1
        t6 = t2 * t5       # Final multiplication
        return t6

# Initialize the model
model = NewModel()

# Generate an input tensor of shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 128, 128)  # Example input tensor

# Get the model output with the input tensor
output = model(input_tensor)

# Display the output shape
print("Output shape:", output.shape)
