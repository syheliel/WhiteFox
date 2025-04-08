import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)  # Pointwise convolution with kernel size 1

    def forward(self, x1):
        t1 = self.conv(x1)               # Apply pointwise convolution
        t2 = t1 * 0.5                    # Multiply by 0.5
        t3 = t1 * t1                     # Square the output of the convolution
        t4 = t3 * t1                     # Cube the output of the convolution
        t5 = t4 * 0.044715               # Multiply by 0.044715
        t6 = t1 + t5                     # Add the output of the convolution to t5
        t7 = t6 * 0.7978845608028654     # Multiply by 0.7978845608028654
        t8 = torch.tanh(t7)              # Apply the hyperbolic tangent function
        t9 = t8 + 1                      # Add 1 to the output of the tanh
        t10 = t2 * t9                    # Multiply t2 by t9
        return t10

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 color channels, 64x64 image size
output = model(input_tensor)               # Forward pass
