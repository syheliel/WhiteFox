import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a convolution layer with weight tensor (8 filters, 3 channels, kernel size 3)
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        # Define a scalar tensor (float) that will be used in the binary operation
        self.scalar = torch.tensor(2.0, dtype=torch.float32)

    def forward(self, x):
        # Apply convolution to the input tensor
        t1 = self.conv(x)
        # Apply a binary operation (addition) with the output of the convolution and the scalar attribute
        t2 = t1 + self.scalar
        return t2

# Initializing the model
model = CustomModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image size
output_tensor = model(input_tensor)

# Show the output shape
print(output_tensor.shape)
