import torch

# Define the model
class SimpleReLUModel(torch.nn.Module):
    def __init__(self):
        super(SimpleReLUModel, self).__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.relu(t1)  # Apply ReLU activation
        return t2

# Initialize the model
model = SimpleReLUModel()

# Generate input tensor for the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Pass the input tensor through the model
output_tensor = model(input_tensor)

# Display the output shape
print(f"Output shape: {output_tensor.shape}")
