import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self, dim=1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # Unbind the input tensor along the specified dimension
        unbound_tensors = torch.unbind(input_tensor, dim=self.dim)
        # Retrieve the i-th tensor from the unbound tensors
        i = 0  # For example, we take the first tensor
        t2 = unbound_tensors[i]
        # Stack the tensors along the specified dimension
        t3 = torch.stack(unbound_tensors, dim=self.dim)
        return t3

# Initializing the model
m = Model(dim=1)

# Inputs to the model
# Creating an input tensor of shape (batch_size, channels, height, width)
input_tensor = torch.randn(2, 3, 64, 64)  # Example input tensor
output = m(input_tensor)

# Output shape
print("Output shape:", output.shape)
