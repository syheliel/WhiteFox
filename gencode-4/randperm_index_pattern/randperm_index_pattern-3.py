import torch

class ShuffleModel(torch.nn.Module):
    def __init__(self, slice_shape):
        super(ShuffleModel, self).__init__()
        self.slice_shape = slice_shape

    def forward(self, x):
        # Generate a random permutation of integers from 0 to x.shape[0] and slice it
        index = torch.randperm(x.shape[0], device=x.device)[:self.slice_shape]
        # Index the input tensor x with the generated index
        output = x[index]
        return output, index

# Initializing the model with a specified slice shape
slice_shape = 5  # Example slice shape
model = ShuffleModel(slice_shape)

# Inputs to the model
input_tensor = torch.randn(10, 3, 64, 64)  # Example input tensor with shape (10, 3, 64, 64)

# Forward pass through the model
output_tensor, indices = model(input_tensor)

# Print the output shape and indices used for shuffling
print("Output shape:", output_tensor.shape)
print("Indices used for shuffling:", indices)
