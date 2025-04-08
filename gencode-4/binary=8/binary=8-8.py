import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, other):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 + other  # Add another tensor to the output of the convolution
        return t2

# Initializing the model
model = Model()

# Generating an input tensor of shape (batch_size=1, channels=3, height=64, width=64)
input_tensor = torch.randn(1, 3, 64, 64)

# Generating the 'other' tensor of the same shape as the output of the convolution
# The output shape will be (1, 8, 64, 64) because the convolution has 8 output channels
other_tensor = torch.randn(1, 8, 64, 64)

# Forward pass through the model
output = model(input_tensor, other_tensor)

# Display the output shape
print(output.shape)  # Expected output shape: (1, 8, 64, 64)
