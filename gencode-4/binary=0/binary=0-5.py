import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, other_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 + other_tensor         # Add another tensor to the output of the convolution
        return t2

# Initializing the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 color channels, 64x64 image
other_tensor = torch.randn(1, 16, 64, 64)  # Another tensor of the same shape as the output of the conv layer

# Getting the output from the model
output = model(input_tensor, other_tensor)

# Display the output shape
print(output.shape)
