import torch

class ShuffleModel(torch.nn.Module):
    def __init__(self, slice_shape):
        super().__init__()
        self.slice_shape = slice_shape
    
    def forward(self, x):
        # Generate a random permutation of indices
        index = torch.randperm(x.shape[0], device=x.device)[:self.slice_shape]
        # Index the input tensor using the generated indices
        output = x[index]
        return output, index

# Initializing the model with a specific slice shape
slice_shape = 5
model = ShuffleModel(slice_shape)

# Generating the input tensor
input_tensor = torch.randn(10, 3, 64, 64)  # Example input tensor with shape (10, 3, 64, 64)

# Forward pass through the model
output, indices = model(input_tensor)
