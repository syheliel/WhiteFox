import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a pointwise transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)
        self.negative_slope = 0.01  # Leaky ReLU negative slope

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply transposed convolution
        t2 = t1 > 0  # Create mask where output is greater than 0
        t3 = t1 * self.negative_slope  # Multiply output by negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the mask
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)  # Input tensor with shape (N, C, H, W)
output = m(x1)

# Display the output shape
print(output.shape)
