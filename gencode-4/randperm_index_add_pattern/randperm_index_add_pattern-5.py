import torch

class IndexAddModel(torch.nn.Module):
    def __init__(self, x_size, y_size):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size

    def forward(self, x, y):
        # Generate a random permutation of integers from 0 to x.shape[0] and select the first y.shape[0] elements
        index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
        # Add the elements of y to the elements of x at the positions specified by index along dimension 0
        result = torch.index_add(x, dim=0, source=y, index=index)
        return result, index

# Model initialization
x_size = (10, 5)  # Example shape for x
y_size = (3, 5)   # Example shape for y
model = IndexAddModel(x_size, y_size)

# Inputs to the model
x = torch.randn(x_size)  # Random tensor for x with shape (10, 5)
y = torch.randn(y_size)  # Random tensor for y with shape (3, 5)

# Forward pass
result, index = model(x, y)

print("Result:\n", result)
print("Index:\n", index)
