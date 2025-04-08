import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = torch.sigmoid(t1)         # Apply the sigmoid function
        t3 = t1 * t2                   # Multiply the output of the transposed convolution by the output of the sigmoid function
        return t3

# Initializing the model
model = Model()

# Generating input tensor
# The input tensor shape should match the output of a previous layer, for instance:
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Getting the output from the model
output = model(input_tensor)

# Display the output shape
print(output.shape)
