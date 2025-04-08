import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)
        self.negative_slope = 0.01  # Typical value for Leaky ReLU

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 > 0  # Create a mask where the output of the convolution is greater than 0
        t3 = t1 * self.negative_slope  # Multiply the output of the convolution by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the mask to the output of the convolution
        return t4

# Initialize the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Get the output from the model using the input tensor
output_tensor = model(input_tensor)

# Print the output shape
print(f"Output tensor shape: {output_tensor.shape}")
