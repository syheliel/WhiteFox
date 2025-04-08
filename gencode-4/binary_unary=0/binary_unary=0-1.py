import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.other_tensor = other_tensor

    def forward(self, x1):
        t1 = self.conv(x1)               # Apply pointwise convolution
        t2 = t1 + self.other_tensor      # Add another tensor to the output
        t3 = torch.nn.functional.relu(t2) # Apply ReLU activation
        return t3

# Initializing the model with a random tensor to add
other_tensor = torch.randn(1, 8, 64, 64)  # Must match the output shape of the convolution
m = Model(other_tensor)

# Input to the model
x1 = torch.randn(1, 3, 64, 64)  # Random input tensor with shape [batch_size, channels, height, width]
output = m(x1)

# Output the result
print(output)
