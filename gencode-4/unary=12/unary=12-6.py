import torch

class GatedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution layer with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Apply sigmoid function
        t2 = torch.sigmoid(t1)
        # Multiply the output of the convolution by the output of the sigmoid function
        t3 = t1 * t2
        return t3

# Initialize the model
model = GatedConvModel()

# Input tensor for the model
input_tensor = torch.randn(1, 3, 64, 64)

# Get the output of the model
output = model(input_tensor)

# Display output shape
print(output.shape)
