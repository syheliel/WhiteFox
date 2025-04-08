import torch

class UnbindModel(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # Unbind the input tensor along the specified dimension
        unbound_tensors = torch.unbind(input_tensor, dim=self.dim)
        
        # Get the first tensor from the unbound tensors
        first_tensor = unbound_tensors[0]
        
        # Stack the tensors along the specified dimension
        stacked_tensor = torch.stack(unbound_tensors, dim=self.dim)
        
        return stacked_tensor

# Initializing the model
model = UnbindModel(dim=1)

# Inputs to the model
input_tensor = torch.randn(5, 3, 64, 64)  # Example input tensor with shape [5, 3, 64, 64]
output = model(input_tensor)

# Printing the output shape
print(output.shape)  # The expected output shape will be [5, 5, 3, 64, 64]
