import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)
 
    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply transposed convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1 to the error function output
        t6 = t2 * t5  # Multiply by the modified error function output
        return t6

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)  # Adjusting input channels to match the transposed conv layer
output = model(x1)

# Print output shape
print(output.shape)
