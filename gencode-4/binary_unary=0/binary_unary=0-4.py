import torch
import torch.nn as nn

# Define the model
class MyModel(nn.Module):
    def __init__(self, other_tensor_shape):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        # Initialize the other tensor with the shape
        self.other = torch.randn(other_tensor_shape)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 + self.other  # Add another tensor to the output of the convolution
        t3 = torch.relu(t2)  # Apply the ReLU activation function
        return t3

# Initialize the model
other_tensor_shape = (1, 8, 64, 64)  # Shape must match the output of the convolution
model = MyModel(other_tensor_shape)

# Create input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Pass the input tensor through the model
output = model(input_tensor)

# Print output shape
print(output.shape)
