import torch

class NewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using a pointwise convolution with different parameters
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Increased output channels

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution with kernel size 1
        t2 = t1 * 0.5  # Multiply the output of the convolution by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply the output of the convolution by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function to the output of the convolution
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the output of the convolution by the output of the error function
        return t6

# Initializing the model
new_model = NewModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 128, 128)  # Different input size
output = new_model(input_tensor)
