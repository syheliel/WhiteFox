import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1) # Apply pointwise transposed convolution
        t2 = t1 + 3                   # Add 3 to the output of the transposed convolution
        t3 = torch.clamp_min(t2, 0)  # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp the output to a maximum of 6
        t5 = t1 * t4                  # Multiply the output of the transposed convolution by the clamped output
        t6 = t5 / 6                   # Divide the output of the multiplication by 6
        return t6

# Initializing the model
model = Model()

# Generate input tensor for the model
# Assuming input tensor shape is (batch_size, channels, height, width)
input_tensor = torch.randn(1, 8, 64, 64)  # Example input tensor with 8 channels

# Get the output from the model
output_tensor = model(input_tensor)

# Display the shape of the output tensor
print("Output tensor shape:", output_tensor.shape)
