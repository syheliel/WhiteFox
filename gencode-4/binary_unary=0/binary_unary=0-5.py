import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.other = other_tensor  # Store the additional tensor

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 + self.other  # Add the other tensor
        t3 = torch.relu(t2)  # Apply ReLU activation
        return t3

# Initialize the other tensor with the same shape as the output of the convolution
other_tensor = torch.randn(1, 8, 64, 64)  # Assuming input tensor is (1, 3, 64, 64)

# Initialize the model with the other tensor
model = Model(other_tensor)

# Create the input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Get the output by passing the input tensor through the model
output = model(input_tensor)

# Print the output shape
print(output.shape)  # Should be (1, 8, 64, 64)
