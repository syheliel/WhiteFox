import torch

class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.other = other_tensor  # This tensor will be added to the output of the convolution
    
    def forward(self, x):
        t1 = self.conv(x)          # Apply pointwise convolution
        t2 = t1 + self.other       # Add another tensor to the output of the convolution
        t3 = torch.relu(t2)        # Apply the ReLU activation function
        return t3

# Initialize the other tensor that will be added
other_tensor = torch.randn(1, 8, 64, 64)  # Matching the output shape of the convolution

# Initialize the model with the other tensor
model = Model(other_tensor)

# Generate an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Get the model's output
output = model(input_tensor)

# Print the output shape
print(output.shape)  # Expected output shape: (1, 8, 64, 64)
