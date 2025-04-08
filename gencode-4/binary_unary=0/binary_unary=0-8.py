import torch

class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.other = other_tensor

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 + self.other           # Add another tensor to the output of the convolution
        t3 = torch.relu(t2)            # Apply the ReLU activation function
        return t3

# Example of initializing the model with an additional tensor
other_tensor = torch.randn(1, 8, 64, 64)  # Same shape as the output of the convolution
model = Model(other_tensor)

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

print(output.shape)  # Should print the shape of the output tensor
