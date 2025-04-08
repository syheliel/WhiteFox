import torch

# Model
class RandomIndexAddModel(torch.nn.Module):
    def __init__(self, x_size, y_size):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size

    def forward(self, x, y):
        # Generate a random permutation of integers from 0 to x.shape[0]
        index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
        # Perform index addition
        result = torch.index_add(x, dim=0, source=y, index=index)
        return result, index

# Initializing the model
# Example sizes for x and y
x_size = (10, 5)  # 10 rows and 5 features
y_size = (3, 5)   # 3 rows and 5 features to add
model = RandomIndexAddModel(x_size, y_size)

# Inputs to the model
x = torch.randn(x_size)  # Input tensor x
y = torch.randn(y_size)  # Input tensor y

# Getting the output from the model
result, index = model(x, y)

print("Result:\n", result)
print("Index:\n", index)
