import torch

class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a transposed convolution layer with input channels of 8 and output channels of 3
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution to the input tensor
        t2 = torch.relu(t1)           # Apply the ReLU activation function to the output of the transposed convolution
        return t2

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
# Create an input tensor with 1 sample, 8 channels, and 64x64 spatial dimensions
input_tensor = torch.randn(1, 8, 64, 64)
output = model(input_tensor)

# Display the shape of the output tensor
print(output.shape)
