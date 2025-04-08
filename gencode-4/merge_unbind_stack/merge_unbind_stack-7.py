import torch

# Define the model class
class UnbindModel(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # Unbind the input tensor along the specified dimension
        t1 = torch.unbind(input_tensor, dim=self.dim)
        # Get the i-th tensor from the unbound tensors (for example, index 0)
        t2 = t1[0]
        # Stack the tensors along the specified dimension
        t3 = torch.stack(t1, dim=self.dim)  # Alternatively, you can use torch.cat(t1, dim=self.dim)
        return t3

# Initialize the model
model = UnbindModel(dim=1)

# Create the input tensor
input_tensor = torch.randn(2, 3, 64, 64)  # Example input tensor with shape (2, 3, 64, 64)

# Forward pass through the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
