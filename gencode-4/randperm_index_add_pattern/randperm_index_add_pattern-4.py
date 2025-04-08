import torch

# Model definition
class IndexAddModel(torch.nn.Module):
    def __init__(self, input_size, add_size):
        super(IndexAddModel, self).__init__()
        self.input_size = input_size
        self.add_size = add_size
        self.x = torch.randn(input_size)  # Initialize x with random values

    def forward(self, y):
        # Generate a random permutation of integers from 0 to x.shape[0]
        index = torch.randperm(self.x.shape[0], device=self.x.device)[:y.shape[0]]
        # Add the elements of y to the elements of x at the positions specified by index
        result = torch.index_add(self.x, dim=0, source=y, index=index)
        return result, index

# Initializing the model
input_size = 10  # Size of the initial tensor x
add_size = 3     # Size of the tensor y to be added
model = IndexAddModel(input_size, add_size)

# Inputs to the model
y = torch.randn(add_size)  # Random tensor y to be added
output, index = model(y)

# Printing the output and index
print("Output:", output)
print("Index:", index)

y = torch.randn(add_size)  # Random tensor y to be added
