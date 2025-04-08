import torch

class RandomIndexAddModel(torch.nn.Module):
    def __init__(self, input_size, add_size):
        super(RandomIndexAddModel, self).__init__()
        self.input_size = input_size
        self.add_size = add_size

    def forward(self, x, y):
        # Generate a random permutation of integers from 0 to x.shape[0] and select the first y.shape[0] elements
        index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
        
        # Add the elements of y to the elements of x at the positions specified by index along dimension 0
        result = torch.index_add(x, dim=0, source=y, index=index)
        
        return result, index

# Initialize the model
input_size = 10  # Size of the input tensor (number of rows)
add_size = 3     # Size of the tensor to add
model = RandomIndexAddModel(input_size, add_size)

# Inputs to the model
x = torch.randn(input_size, 5)  # Input tensor with shape (10, 5)
y = torch.randn(add_size, 5)     # Tensor to add with shape (3, 5)

# Perform the forward pass
result, index = model(x, y)

# Output the results
print("Result Tensor:\n", result)
print("Index Tensor:\n", index)
