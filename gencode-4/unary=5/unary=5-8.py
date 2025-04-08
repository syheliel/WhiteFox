import torch

# Define the Model
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution: 8 output channels, 3 input channels, kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5                  # Multiply the output of the transposed convolution by 0.5
        t3 = t1 * 0.7071067811865476   # Multiply the output of the transposed convolution by 0.7071067811865476
        t4 = torch.erf(t3)             # Apply the error function to the output of the transposed convolution
        t5 = t4 + 1                    # Add 1 to the output of the error function
        t6 = t2 * t5                   # Multiply the output of the transposed convolution by the output of the error function
        return t6

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)  # Input tensor with shape (batch_size, channels, height, width)
output = model(x1)

# Print the output shape
print(output.shape)
