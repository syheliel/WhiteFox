import torch

# Define the model
class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self, input1_shape, input2_shape):
        super().__init__()
        self.input1_shape = input1_shape
        self.input2_shape = input2_shape

    def forward(self, input1, input2):
        output = torch.mm(input1, input2)
        return output

# Initialize the model with specific input shapes
# The first input tensor must have a first dimension >= 10240
# The second input tensor must have dimensions < 32
model = MatrixMultiplicationModel(input1_shape=(10240, 64), input2_shape=(64, 32))

# Generate input tensors
# Both input tensors must be on the same device (CPU in this case)
input1 = torch.randn(10240, 64)  # First input tensor (2D)
input2 = torch.randn(64, 32)      # Second input tensor (2D)

# Forward pass through the model
output = model(input1, input2)

# Check the output shape
print(output.shape)  # This should print: torch.Size([10240, 32])
