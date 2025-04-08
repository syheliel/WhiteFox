import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, other_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 + other_tensor  # Add another tensor to the output of the convolution
        return t2

# Initializing the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, height and width of 64
other_tensor = torch.randn(1, 8, 64, 64)  # Must match the output shape of the convolution (1, 8, 64, 64)

# Forward pass
output = model(input_tensor, other_tensor)

# Display output shape
print(output.shape)
