import torch

class AddModel(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.other_tensor = other_tensor  # Store the tensor to add

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 + self.other_tensor  # Add the other tensor to the convolution output
        return t2

# Example of creating a tensor to be added
other_tensor = torch.randn(1, 8, 64, 64)  # Make sure the dimensions match the output of the convolution

# Initializing the model with the other tensor
model = AddModel(other_tensor)

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # A random input tensor
output = model(input_tensor)

# Print the output shape
print(output.shape)
