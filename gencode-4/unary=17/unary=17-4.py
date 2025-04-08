import torch

class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a pointwise transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.relu(t1)          # Apply ReLU activation
        return t2

# Initialize the model
model = TransposeConvModel()

# Generate an input tensor with appropriate dimensions
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Get the model output
output_tensor = model(input_tensor)

# Print output tensor shape
print("Output tensor shape:", output_tensor.shape)
