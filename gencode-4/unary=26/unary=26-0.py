import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1, input channels 8, output channels 3
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)
        self.negative_slope = 0.1  # Example negative slope for the Leaky ReLU

    def forward(self, x1):
        t1 = self.conv_transpose(x1)
        t2 = t1 > 0  # Create a mask where the output of the convolution is greater than 0
        t3 = t1 * self.negative_slope  # Multiply the output of the convolution by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the mask to the output of the convolution
        return t4

# Initializing the model
model = Model()

# Generating input tensor for the model
# Input tensor should have 8 channels (to match the output channels of the previous operation)
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimension

# Running the model with the input tensor
output = model(input_tensor)

# Output
print(output.shape)  # Should be (1, 3, 64, 64)
