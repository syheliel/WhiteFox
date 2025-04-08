import torch

# Define the model
class NewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply the output of the convolution by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply the output of the convolution by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function to the output of the convolution
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the result of t2 by t5
        return t6

# Initialize the model
model = NewModel()

# Create an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image size

# Forward pass through the model
output_tensor = model(input_tensor)

# Output the shape of the output tensor
print(f"Output tensor shape: {output_tensor.shape}")
