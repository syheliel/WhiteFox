import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, input_tensor, other):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 + other  # Add another tensor to the output of the convolution
        return t2

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Input tensor with shape (batch_size, channels, height, width)
other_tensor = torch.randn(1, 8, 64, 64)  # Other tensor with the same shape as the output of the convolution
output = m(input_tensor, other_tensor)

# Printing the output shape
print(output.shape)
