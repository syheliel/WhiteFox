import torch

class NewModel(torch.nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        # Pointwise convolution with a different configuration (e.g., 3 input channels, 16 output channels)
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the output of the convolution by the output of the error function
        return t6

# Initializing the new model
new_model = NewModel()

# Generating an input tensor
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 color channels, 128x128 image

# Forward pass through the model
output_tensor = new_model(input_tensor)

# Output shape
print("Output shape:", output_tensor.shape)
