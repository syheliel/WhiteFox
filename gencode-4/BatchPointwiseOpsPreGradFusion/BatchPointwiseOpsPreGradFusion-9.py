import torch

# Define a new model class
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using a pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Changed the output channels to 16

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the output of the convolution by the result
        return t6

# Initialize the model
model = CustomModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 128, 128)  # Example input tensor with different size

# Pass the input tensor through the model
output_tensor = model(input_tensor)

# Output to verify
print(output_tensor.shape)  # Should give the shape of the output tensor
