import torch

# Define the model
class TransposedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using ConvTranspose2d with input channels=8 and output channels=3, kernel size=1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)
 
    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5                  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476   # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)             # Apply the error function
        t5 = t4 + 1                     # Add 1
        t6 = t2 * t5                    # Multiply the two results
        return t6

# Initializing the model
model = TransposedModel()

# Generating input tensor
# The input tensor should have 8 channels (to match the output of the previous layer), with a height and width of 64
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Forward pass through the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
