import torch

# Define the model
class IndexAddModel(torch.nn.Module):
    def __init__(self, x_shape, y_shape):
        super().__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape

    def forward(self, x, y):
        # Generate a random permutation of integers from 0 to x.shape[0]
        index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
        
        # Add the elements of y to the elements of x at the positions specified by index along dimension 0
        result = torch.index_add(x, dim=0, source=y, index=index)
        
        return result, index

# Initialize the model with specific input shapes
x_shape = (10, 3)  # For example, x has shape (10, 3)
y_shape = (5, 3)   # For example, y has shape (5, 3)
model = IndexAddModel(x_shape, y_shape)

# Generate input tensors
x = torch.randn(x_shape)  # Random tensor of shape (10, 3)
y = torch.randn(y_shape)  # Random tensor of shape (5, 3)

# Forward pass
result, index = model(x, y)

# Output the results
print("Result:", result)
print("Index:", index)
