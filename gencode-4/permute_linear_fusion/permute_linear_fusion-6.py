import torch

class PermuteLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Assume we want to apply a linear layer after permutation
        self.linear = torch.nn.Linear(64, 128)  # Linear layer expects input of size 64

    def forward(self, input_tensor):
        # Permute the last two dimensions of the input tensor
        t1 = input_tensor.permute(0, 2, 1)  # Swap the last two dimensions
        # Apply linear transformation to the permuted tensor
        t2 = self.linear(t1)  # Linear expects the last dimension to match the input size of the linear layer
        return t2

# Initializing the model
model = PermuteLinearModel()

# Generating input tensor
# The input tensor should have at least 3 dimensions; here we use a batch size of 1, height of 32, and width of 64
input_tensor = torch.randn(1, 32, 64)  # Shape: (batch_size, height, width)

# Passing the input tensor through the model
output = model(input_tensor)
print(output.shape)  # This will show the shape of the output tensor after the linear transformation
