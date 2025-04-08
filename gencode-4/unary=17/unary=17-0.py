import torch

# Define the model
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.relu(t1)           # Apply ReLU activation function
        return t2

# Initialize the model
model = TransposeConvModel()

# Create an input tensor
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Run the model with the input tensor
output = model(input_tensor)

# Output the shape of the output tensor
print("Output shape:", output.shape)  # Expected shape: (1, 3, 64, 64)
