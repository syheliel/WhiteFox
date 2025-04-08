import torch

# Model Definition
class NewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using a different configuration for the convolutional layer
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)  # 3 input channels, 16 output channels

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution with kernel size 1
        t2 = t1 * 0.5  # Multiply the output of the convolution by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply the output of the convolution by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function to the output of the convolution
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the output of the convolution by the output of the error function
        return t6

# Initializing the model
model = NewModel()

# Generating an input tensor for the model
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 color channels, 128x128 image size

# Forward pass through the model
output = model(input_tensor)

# Display the output shape
print("Output shape:", output.shape)
