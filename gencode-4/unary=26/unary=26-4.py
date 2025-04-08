import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 > 0  # Create a mask where the output of the convolution is greater than 0
        negative_slope = 0.01  # Define the negative slope
        t3 = t1 * negative_slope  # Multiply the output of the convolution by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the mask to the output of the convolution
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)  # Example input tensor
__output__ = m(x1)

# Print output shape for verification
print(__output__.shape)
