import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Using a different number of input and output channels
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)  
        # Multiply the output of the convolution by 0.5
        t2 = t1 * 0.5  
        # Multiply the output of the convolution by 0.7071067811865476
        t3 = t1 * 0.7071067811865476  
        # Apply the error function
        t4 = torch.erf(t3)  
        # Add 1 to the output of the error function
        t5 = t4 + 1  
        # Multiply the output of the convolution by the output of the error function
        t6 = t2 * t5  
        return t6

# Initializing the model
model = MyModel()

# Input tensor for the model
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image

# Forward pass
output = model(input_tensor)

# Print output shape
print(output.shape)
