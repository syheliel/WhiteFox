import torch

class TransposedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.sigmoid(t1)        # Apply the sigmoid function
        t3 = t1 * t2                  # Multiply the output of the transposed convolution by the output of the sigmoid function
        return t3

# Initializing the model
model = TransposedModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 spatial dimensions
output_tensor = model(input_tensor)

# Printing the output shape to verify
print("Output shape:", output_tensor.shape)
