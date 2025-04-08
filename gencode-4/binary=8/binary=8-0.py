import torch

class AddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)

    def forward(self, input_tensor, other):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 + other               # Add the other tensor to the output of the convolution
        return t2

# Initializing the model
model = AddModel()

# Generating input tensor and the 'other' tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Input tensor with shape (batch_size, channels, height, width)
other_tensor = torch.randn(1, 8, 64, 64)   # 'Other' tensor must have the same shape as the convolution output

# Forward pass
output = model(input_tensor, other_tensor)
