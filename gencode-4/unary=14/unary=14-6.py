import torch

# Model
class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = torch.sigmoid(t1)        # Apply the sigmoid function
        t3 = t1 * t2                  # Multiply the output of the transposed convolution by the output of the sigmoid function
        return t3

# Initializing the model
model = ModelTransposeConv()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, height and width of 64
output = model(input_tensor)

# Print the output shape
print("Output shape:", output.shape)
