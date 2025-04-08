import torch

# Model Definition
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.relu(t1)           # Apply the ReLU activation function
        return t2

# Initializing the model
model = TransposedConvModel()

# Inputs to the model
input_tensor = torch.randn(1, 8, 32, 32)  # Example input tensor with shape (batch_size, channels, height, width)

# Forward pass
output = model(input_tensor)

# Print output shape
print("Output shape:", output.shape)
