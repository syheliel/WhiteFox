import torch

# Model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1)  # Pointwise convolution
        self.other_tensor = torch.randn(1, 8, 64, 64)  # Another tensor to add

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 + self.other_tensor  # Add another tensor
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initializing the model
model = CustomModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 input channels, 64x64 image size
output = model(input_tensor)

# Output the shape of the result
print(output.shape)
