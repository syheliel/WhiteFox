import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.other = other_tensor

    def forward(self, x1):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = t1 + self.other  # Add another tensor to the output of the convolution
        t3 = torch.relu(t2)  # Apply ReLU activation
        return t3

# Initializing the model
# Create an example 'other' tensor with the same shape as the output of the convolution
other_tensor = torch.randn(1, 16, 64, 64)  # Batch size of 1, 16 output channels, 64x64 spatial dimensions
model = Model(other_tensor)

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 input channels, 64x64 spatial dimensions
output = model(input_tensor)

# Printing the output shape
print(output.shape)
