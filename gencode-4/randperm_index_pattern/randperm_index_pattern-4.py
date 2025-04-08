import torch

class ShuffleModel(torch.nn.Module):
    def __init__(self, slice_shape):
        super().__init__()
        self.slice_shape = slice_shape

    def forward(self, x):
        # Generate a random permutation of indices
        index = torch.randperm(x.shape[0], device=x.device)[:self.slice_shape]
        # Index the input tensor with the generated random indices
        output = x[index]
        return output, index

# Initializing the model with a specific slice shape
slice_shape = 5  # For example, we want to get 5 elements
model = ShuffleModel(slice_shape)

# Generating an input tensor
input_tensor = torch.randn(10, 3, 64, 64)  # A tensor with 10 samples, 3 channels, 64x64 size

# Forward pass
shuffled_output, indices = model(input_tensor)

# Print the results
print("Shuffled Output Shape:", shuffled_output.shape)
print("Indices Used for Shuffling:", indices)
