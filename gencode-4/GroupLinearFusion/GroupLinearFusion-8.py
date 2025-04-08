import torch

class AnotherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Pointwise convolution with kernel size 1

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply the output of the convolution by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply the output of the convolution by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function to the output of the convolution
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the output of the convolution by the output of the error function
        return t6

# Initializing the model
model = AnotherModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 128, 128)  # Example input tensor with shape (batch_size, channels, height, width)

# Forward pass through the model
output = model(input_tensor)

input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 input channels, 128 height, and 128 width
