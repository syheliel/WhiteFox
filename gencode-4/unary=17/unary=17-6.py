import torch

# Model definition
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.nn.functional.relu(t1)  # Apply ReLU activation
        return t2

# Initializing the model
model = TransposedConvModel()

# Generate an input tensor for the model
# Assuming the input to the transposed convolution has 8 channels and spatial dimensions of 64x64
input_tensor = torch.randn(1, 8, 64, 64)

# Get the output of the model
output = model(input_tensor)

# Print the output shape
print(f"Output shape: {output.shape}")
