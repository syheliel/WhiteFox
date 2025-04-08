import torch

# Model
class ShuffleModel(torch.nn.Module):
    def __init__(self, slice_shape: int):
        super().__init__()
        self.slice_shape = slice_shape

    def forward(self, x):
        # Generate a random permutation of integers from 0 to x.shape[0]
        index = torch.randperm(x.shape[0], device=x.device)[:self.slice_shape]
        # Index the input tensor x with the generated index
        output = x[index]
        return output, index

# Initializing the model with a specific slice_shape
slice_shape = 5  # Define the number of elements to shuffle
m = ShuffleModel(slice_shape)

# Inputs to the model
input_shape = (10, 3, 64, 64)  # Example input shape
x = torch.randn(input_shape)

# Forward pass
__output__, indices = m(x)

# Output results
print("Shuffled Output Shape:", __output__.shape)
print("Indices Used for Shuffling:", indices)
