import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, other_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution with kernel size 1 to the input tensor
        t2 = t1 + other_tensor  # Add another tensor to the output of the convolution
        return t2

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 input channels, 64x64 image
other_tensor = torch.randn(1, 8, 64, 64)  # Same spatial dimensions as the output of the conv layer

# Forward pass
output = model(input_tensor, other_tensor)

# Print the shape of the output
print(output.shape)  # Should be [1, 8, 64, 64]
