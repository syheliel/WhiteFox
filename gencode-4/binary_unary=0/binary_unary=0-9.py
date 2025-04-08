import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, other):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 + other                # Add another tensor to the output of the convolution
        t3 = torch.relu(t2)           # Apply ReLU activation
        return t3

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)   # Input tensor of shape (batch_size, channels, height, width)
other_tensor = torch.randn(1, 8, 64, 64)    # Another tensor to be added, must match the output shape of conv layer

# Forward pass
output = model(input_tensor, other_tensor)

# Print the output shape
print(output.shape)
