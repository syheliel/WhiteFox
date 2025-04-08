import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution: input channels = 8, output channels = 3, kernel size = 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)           # Apply pointwise transposed convolution
        t2 = torch.sigmoid(t1)                 # Apply the sigmoid function
        return t2

# Initializing the model
model = TransposeConvModel()

# Generating inputs to the model
# Input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 8, 64, 64)  # Example input tensor

# Getting the output from the model
output_tensor = model(input_tensor)

# Print the shape of the output tensor
print("Output tensor shape:", output_tensor.shape)
