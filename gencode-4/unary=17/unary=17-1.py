import torch

# Define the model
class TransposeConvReLUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.nn.functional.relu(t1)  # Apply the ReLU activation function
        return t2

# Initializing the model
model = TransposeConvReLUModel()

# Inputs to the model
input_tensor = torch.randn(1, 8, 64, 64)  # Input tensor with 8 channels
output_tensor = model(input_tensor)

# Output shape
print(output_tensor.shape)
