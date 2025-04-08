import torch

# Define the model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a convolutional layer with weights and optional bias
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Define an attribute scalar for the binary operation
        self.scalar = torch.tensor(2.0)

    def forward(self, input_tensor):
        # Apply convolution to the input tensor
        t1 = self.conv(input_tensor)
        # Apply binary operation (addition) with the output of the convolution and a scalar
        t2 = t1 + self.scalar
        return t2

# Initialize the model
model = CustomModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Get the output of the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
