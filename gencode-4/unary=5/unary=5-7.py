import torch

# Define the model
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution: input channels = 8, output channels = 3, kernel size = 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5                  # Multiply the output of the transposed convolution by 0.5
        t3 = t1 * 0.7071067811865476   # Multiply the output of the transposed convolution by 0.7071067811865476
        t4 = torch.erf(t3)             # Apply the error function to the output of the transposed convolution
        t5 = t4 + 1                     # Add 1 to the output of the error function
        t6 = t2 * t5                    # Multiply the output of the transposed convolution by the output of the error function
        return t6

# Initialize the model
model = TransposedConvModel()

# Create an input tensor
# Assuming we want to use an input with 8 channels and a spatial size of 64x64
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Forward pass through the model
output = model(input_tensor)
print(output.shape)  # Print the shape of the output tensor
