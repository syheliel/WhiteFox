import torch

# Define the model class
class NewModel(torch.nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1
        t6 = t2 * t5  # Multiply the two results
        return t6

# Initialize the model
model = NewModel()

# Generate input tensor
# Input shape: (batch_size, channels, height, width)
input_tensor = torch.randn(1, 1, 32, 32)  # Batch size of 1, 1 channel, 32x32 image

# Forward pass through the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
