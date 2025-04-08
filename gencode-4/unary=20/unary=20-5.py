import torch

# Model definition
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super(TransposedConvModel, self).__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.sigmoid(t1)        # Apply the sigmoid function
        return t2

# Initializing the model
model = TransposedConvModel()

# Inputs to the model
input_tensor = torch.randn(1, 16, 64, 64)  # Batch size of 1, 16 channels, 64x64 spatial dimensions
output_tensor = model(input_tensor)

# Display the output shape
print(output_tensor.shape)  # Should output: torch.Size([1, 3, 64, 64]) after the transposed convolution
