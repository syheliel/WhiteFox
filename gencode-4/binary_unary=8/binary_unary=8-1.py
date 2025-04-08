import torch

class NewModel(torch.nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, input_tensor, other_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 + other_tensor  # Add another tensor to the output of the convolution
        t3 = torch.relu(t2)  # Apply the ReLU activation function
        return t3

# Initializing the model
model = NewModel()

# Creating input tensors
input_tensor = torch.randn(1, 3, 64, 64)  # Input tensor with batch size 1, 3 channels, 64x64 dimensions
other_tensor = torch.randn(1, 16, 64, 64)  # Another tensor to add, matching the output size of the convolution

# Forward pass through the model
output = model(input_tensor, other_tensor)
