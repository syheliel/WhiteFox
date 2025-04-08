import torch

class IndexAddModel(torch.nn.Module):
    def __init__(self, x_size, y_size):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size

    def forward(self, x, y):
        # Generate a random permutation of indices
        index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
        # Add the elements of y to the elements of x at the positions specified by index along dimension 0
        result = torch.index_add(x, dim=0, source=y, index=index)
        return result, index

# Initializing the model with specific sizes for x and y
x_size = 10  # Size of tensor x
y_size = 5   # Size of tensor y

model = IndexAddModel(x_size, y_size)

# Inputs to the model
x = torch.randn(x_size, 3)  # x is of size (10, 3)
y = torch.randn(y_size, 3)   # y is of size (5, 3)

# Forward pass
output, indices = model(x, y)

# Output check
print("Output Tensor:\n", output)
print("Indices Used:\n", indices)
