import torch

class NewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply the output of the convolution by 0.5
        t3 = t1 * t1  # Square the output of the convolution
        t4 = t3 * t1  # Cube the output of the convolution
        t5 = t4 * 0.044715  # Multiply the cube by 0.044715
        t6 = t1 + t5  # Add the output of the convolution to t5
        t7 = t6 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t8 = torch.tanh(t7)  # Apply the hyperbolic tangent function
        t9 = t8 + 1  # Add 1 to the output of the hyperbolic tangent function
        t10 = t2 * t9  # Multiply the output of the convolution by the output of the hyperbolic tangent function
        return t10

# Initializing the model
model = NewModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

# Output shape
print(f'Output shape: {output.shape}')
