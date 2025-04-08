import torch

class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5                  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476   # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)             # Apply the error function
        t5 = t4 + 1                     # Add 1
        t6 = t2 * t5                    # Multiply by the output of the error function
        return t6

# Initializing the model
model = ModelTransposeConv()

# Generate input tensor
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions
output_tensor = model(input_tensor)
