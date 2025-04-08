import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Pointwise convolution with kernel size 1

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 - 0.5  # Subtract scalar 0.5 from the output of the convolution
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initialize the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Get the output from the model
output_tensor = model(input_tensor)

# Print the output shape
print("Output shape:", output_tensor.shape)
