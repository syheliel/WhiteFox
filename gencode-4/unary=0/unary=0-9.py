import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
 
    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)               # Apply pointwise convolution with kernel size 1
        t2 = t1 * 0.5                              # Multiply the output of the convolution by 0.5
        t3 = t1 * t1                               # Square the output of the convolution
        t4 = t3 * t1                               # Cube the output of the convolution
        t5 = t4 * 0.044715                         # Multiply the cube of the output of the convolution by 0.044715
        t6 = t1 + t5                               # Add the output of the convolution to the result of the previous operation
        t7 = t6 * 0.7978845608028654               # Multiply the result of the previous operation by 0.7978845608028654
        t8 = torch.tanh(t7)                        # Apply the hyperbolic tangent function to the result of the previous operation
        t9 = t8 + 1                                 # Add 1 to the output of the hyperbolic tangent function
        t10 = t2 * t9                              # Multiply the output of the convolution by the output of the hyperbolic tangent function
        return t10

# Initializing the model
model = CustomModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 128, 128)  # Example input tensor with batch size 1, 3 channels, and 128x128 image size
output = model(input_tensor)
