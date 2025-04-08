import torch

class DifferentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the two results
        return t6

# Initializing the model
different_model = DifferentModel()

# Generating input tensor for the model
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image

# Getting the output from the model
output_tensor = different_model(input_tensor)

# Output the shape of the output tensor
print(output_tensor.shape)
