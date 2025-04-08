import torch

class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a transposed convolution layer with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution to the input tensor
        t2 = torch.sigmoid(t1)       # Apply the sigmoid function to the output of the transposed convolution
        return t2

# Initializing the model
model = TransposeConvModel()

# Generating the input tensor
# The input tensor should have 8 channels, for example, with a spatial dimension of 32x32
input_tensor = torch.randn(1, 8, 32, 32)

# Passing the input tensor through the model
output_tensor = model(input_tensor)

# Display the shape of the output tensor
print("Output tensor shape:", output_tensor.shape)
