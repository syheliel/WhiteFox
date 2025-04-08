import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution
        self.other = torch.randn(1, 8, 64, 64)  # Another tensor to add to the convolution output

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 + self.other  # Add another tensor
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initializing the model
model = CustomModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

print("Output shape:", output.shape)  # Print the shape of the output tensor
