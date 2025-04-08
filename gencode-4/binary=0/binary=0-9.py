import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, other_tensor):
        # Apply pointwise convolution
        t1 = self.conv(input_tensor)
        # Add another tensor to the output of the convolution
        t2 = t1 + other_tensor
        return t2

# Initialize the model
model = Model()

# Generate a random input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 input channels, 64x64 image size

# Generate another tensor to add to the output of the convolution
other_tensor = torch.randn(1, 8, 64, 64)  # Same shape as the output of the convolution

# Calculate the output
output = model(input_tensor, other_tensor)

# Print the output shape
print(output.shape)
