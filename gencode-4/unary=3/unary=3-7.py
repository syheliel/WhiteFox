import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # A different pointwise convolution layer with different input and output channels
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
 
    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution with kernel size 1 to the input tensor
        t2 = t1 * 0.5  # Multiply the output of the convolution by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply the output of the convolution by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function to the output of the convolution
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the output of the convolution by the output of the error function
        return t6

# Initializing the model
model = CustomModel()

# Generate a random input tensor for the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 color channels, 64x64 image size
output = model(input_tensor)

# Print the output shape
print(output.shape)
