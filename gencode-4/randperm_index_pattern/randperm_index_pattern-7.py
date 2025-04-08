import torch

class ShuffleModel(torch.nn.Module):
    def __init__(self, slice_shape):
        super().__init__()
        self.slice_shape = slice_shape

    def forward(self, x):
        # Generate a random permutation of integers from 0 to x.shape[0] and slice it to the shape specified by slice_shape
        index = torch.randperm(x.shape[0], device=x.device)[:self.slice_shape]
        # Index the input tensor x with the generated index
        output = x[index]
        return output, index

# Initializing the model with a specified slice shape
slice_shape = 5  # Example slice shape
model = ShuffleModel(slice_shape)

# Inputs to the model
x = torch.randn(10, 3, 64, 64)  # Random input tensor of shape (10, 3, 64, 64)
output, indices = model(x)

print("Output shape:", output.shape)
print("Indices used for shuffling:", indices)
