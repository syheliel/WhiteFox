import torch

class NewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 16, 1, stride=1, padding=0)  # Different input/output channels and padding

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution with kernel size 1 to the input tensor
        t2 = t1 * 0.5  # Multiply the output of the convolution by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply the output of the convolution by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function to the output of the convolution
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the output of the convolution by the output of the error function
        return t6

# Initializing the new model
new_model = NewModel()

# Generating inputs to the model
input_tensor = torch.randn(1, 4, 128, 128)  # Different input dimensions
output = new_model(input_tensor)

print(output.shape)  # Display the output shape
