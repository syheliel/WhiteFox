import torch

class NewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using a pointwise convolution with 3 input channels and 16 output channels
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply the output by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply the output by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the two results together
        return t6

# Initializing the model
model = NewModel()

# Create an input tensor with a batch size of 1, 3 channels, and 64x64 spatial dimensions
input_tensor = torch.randn(1, 3, 64, 64)

# Pass the input tensor through the model
output_tensor = model(input_tensor)

# Display the output tensor shape
print(output_tensor.shape)
