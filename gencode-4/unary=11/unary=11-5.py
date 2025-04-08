import torch

# Define the model class
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a pointwise transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 + 3                   # Add 3 to the output of the transposed convolution
        t3 = torch.clamp_min(t2, 0)   # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)    # Clamp the output to a maximum of 6
        t5 = t4 / 6                     # Divide by 6
        return t5

# Initialize the model
model = TransposeConvModel()

# Create an input tensor for the model
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Get the output from the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
