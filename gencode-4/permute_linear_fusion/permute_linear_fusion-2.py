import torch

class PermuteLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with input size 16 and output size 32
        self.linear = torch.nn.Linear(16, 32)

    def forward(self, input_tensor):
        # Permute the input tensor to swap the last two dimensions
        # Assuming input_tensor is of shape (batch_size, channels, height, width)
        t1 = input_tensor.permute(0, 2, 3, 1)  # Swap channels with height and width
        # Reshape to (-1, 16) to make it compatible with the linear layer
        t1 = t1.contiguous().view(-1, 16)
        # Apply linear transformation to the permuted tensor
        t2 = self.linear(t1)
        return t2

# Initializing the model
model = PermuteLinearModel()

# Input tensor for the model
# Create a random tensor with shape (1, 4, 4, 4) to meet the requirement of having more than 2 dimensions
input_tensor = torch.randn(1, 4, 4, 4)

# Forward pass through the model
output = model(input_tensor)

# Check the output shape
print(output.shape)
