import torch

# Define the model
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        # Apply transposed convolution
        t1 = self.conv_transpose(x1)
        # Apply sigmoid activation
        t2 = torch.sigmoid(t1)
        return t2

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
# Note: The input size should match the output channels of the transposed convolution layer (8 in this case)
input_tensor = torch.randn(1, 8, 64, 64)

# Forward pass
output = model(input_tensor)

# Print output shape
print("Output shape:", output.shape)
