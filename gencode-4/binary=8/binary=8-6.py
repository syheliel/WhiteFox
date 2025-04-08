import torch

class AddConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution layer
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, other):
        # Apply pointwise convolution
        t1 = self.conv(input_tensor)
        # Add the other tensor to the output of the convolution
        t2 = t1 + other
        return t2

# Initialize the model
model = AddConvModel()

# Create an input tensor of shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 64, 64)

# Create another tensor to add, which should have the same shape as the output of the convolution
# The output shape of the conv layer will be (1, 8, 64, 64)
other = torch.randn(1, 8, 64, 64)

# Run the model with the input tensor and the other tensor
output = model(input_tensor, other)

print(output.shape)  # Output shape will be (1, 8, 64, 64)
