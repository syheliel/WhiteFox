import torch

# Model
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = torch.sigmoid(t1)         # Apply the sigmoid function
        t3 = t1 * t2                   # Multiply the output of the transposed convolution by the output of the sigmoid function
        return t3

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
# Input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 8, 64, 64)  # Example input tensor for the model
output_tensor = model(input_tensor)        # Getting the output from the model

# Display output shape
print(output_tensor.shape)
