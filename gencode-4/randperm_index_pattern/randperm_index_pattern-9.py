import torch

class ShuffleModel(torch.nn.Module):
    def __init__(self, slice_shape):
        super().__init__()
        self.slice_shape = slice_shape

    def forward(self, x):
        # Generate a random permutation of integers from 0 to x.shape[0] and slice it to slice_shape
        index = torch.randperm(x.shape[0], device=x.device)[:self.slice_shape]
        # Index the input tensor x with the generated index
        output = x[index]
        return output, index

# Initializing the model with a specific slice_shape
slice_shape = 5  # Example slice shape
model = ShuffleModel(slice_shape)

# Inputs to the model
# Creating a random input tensor of shape (10, 3, 64, 64)
x = torch.randn(10, 3, 64, 64)

# Forward pass through the model
shuffled_output, indices = model(x)

print("Shuffled Output Shape:", shuffled_output.shape)
print("Indices Used for Shuffling:", indices)
