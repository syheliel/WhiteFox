import torch

# Define the model
class IndexAddModel(torch.nn.Module):
    def __init__(self, input_size, y_size):
        super(IndexAddModel, self).__init__()
        self.input_size = input_size
        self.y_size = y_size

    def forward(self, x, y):
        # Generate a random permutation of integers from 0 to x.shape[0] and select the first y.shape[0] elements
        index = torch.randperm(x.shape[0], device=x.device)[:self.y_size]
        
        # Add elements of y to elements of x at the positions specified by index along dimension 0
        result = torch.index_add(x, dim=0, source=y, index=index)
        
        return result, index

# Initialize the model
input_size = 10   # Size of x
y_size = 5        # Size of y
model = IndexAddModel(input_size, y_size)

# Create input tensors
x = torch.randn(input_size, 3)  # Input tensor x of shape [10, 3]
y = torch.randn(y_size, 3)       # Input tensor y of shape [5, 3]

# Forward pass
output, index = model(x, y)

# Display the output and index
print("Output Tensor:\n", output)
print("Index Tensor:\n", index)
