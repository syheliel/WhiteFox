import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution
        self.other_tensor = torch.randn(1, 8, 64, 64)  # This will be added to the output of the convolution

    def forward(self, x1):
        t1 = self.conv(x1)  # Apply the convolution
        t2 = t1 + self.other_tensor  # Add another tensor
        t3 = torch.relu(t2)  # Apply the ReLU activation function
        return t3

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

# Output shape
print(output.shape)

input_tensor = torch.randn(1, 3, 64, 64)  # Creates a random tensor with the shape (1, 3, 64, 64)
