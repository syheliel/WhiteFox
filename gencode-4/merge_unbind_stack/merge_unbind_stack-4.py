import torch

class UnbindModel(torch.nn.Module):
    def __init__(self, dim=1, index=0):
        super().__init__()
        self.dim = dim
        self.index = index
    
    def forward(self, input_tensor):
        # Unbind the input tensor along the specified dimension
        unbound_tensors = torch.unbind(input_tensor, dim=self.dim)
        
        # Retrieve the i-th tensor from the unbound tensors
        selected_tensor = unbound_tensors[self.index]
        
        # Stack the tensors along the specified dimension
        # Here we stack the tensors from the unbound tensors except the selected one for demonstration
        stacked_tensor = torch.stack([t for i, t in enumerate(unbound_tensors) if i != self.index], dim=self.dim)
        
        return selected_tensor, stacked_tensor

# Initializing the model
model = UnbindModel(dim=1, index=0)

# Inputs to the model
input_tensor = torch.randn(2, 3, 64, 64)  # Example input tensor of shape (2, 3, 64, 64)
output_selected, output_stacked = model(input_tensor)

print("Selected Tensor Shape:", output_selected.shape)
print("Stacked Tensor Shape:", output_stacked.shape)
