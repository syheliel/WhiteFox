import torch

# Define the model
class IndexAddModel(torch.nn.Module):
    def __init__(self, x_size, y_size):
        super(IndexAddModel, self).__init__()
        self.x_size = x_size
        self.y_size = y_size

    def forward(self, x, y):
        # Generate a random permutation of integers from 0 to x.shape[0]
        index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
        # Add the elements of y to the elements of x at the positions specified by index along dimension 0
        result = torch.index_add(x, dim=0, source=y, index=index)
        return result, index

# Initialize the model
x_size = (10, 5)  # 10 rows, 5 features
y_size = (3, 5)   # 3 rows, 5 features
model = IndexAddModel(x_size, y_size)

# Inputs to the model
x = torch.randn(x_size)  # Tensor x with shape (10, 5)
y = torch.randn(y_size)  # Tensor y with shape (3, 5)

# Forward pass
result, index = model(x, y)

# Print output shapes
print("Result shape:", result.shape)
print("Index shape:", index.shape)
