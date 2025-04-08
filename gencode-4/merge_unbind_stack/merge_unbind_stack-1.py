import torch

class Model(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # Unbind the input tensor along the specified dimension
        unbound_tensors = torch.unbind(input_tensor, dim=self.dim)
        
        # Select the i-th tensor from the unbound tensors
        i = 0  # You can change this index to select different tensors
        selected_tensor = unbound_tensors[i]
        
        # Stack the selected tensor along the specified dimension
        stacked_tensor = torch.stack(unbound_tensors, dim=self.dim)
        
        return stacked_tensor

# Initialize the model
m = Model(dim=1)

# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)  # Input tensor of shape (2, 3, 64, 64)
output = m(x1)

# Print the output shape
print(output.shape)
