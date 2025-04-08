import torch

# Define the model
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution: in_channels=8, out_channels=3, kernel_size=1, stride=1, padding=0
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.sigmoid(t1)        # Apply the sigmoid function
        return t2

# Initializing the model
model = TransposedConvModel()

# Inputs to the model
# Let's assume the input tensor has 8 channels (to match the out_channels of the previous layer in a potential pipeline)
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size 1, 8 channels, 64x64 spatial dimensions

# Forward pass
output = model(input_tensor)

# Output shape
print("Output shape:", output.shape)
