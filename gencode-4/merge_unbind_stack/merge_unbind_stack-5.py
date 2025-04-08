import torch

class UnbindConcatModel(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # Unbind the input tensor along the specified dimension
        t1 = torch.unbind(input_tensor, dim=self.dim)
        
        # Get the i-th tensor from the unbound tensors
        i = 0  # You can change this index as needed
        t2 = t1[i]

        # Stack the tensors along the specified dimension
        t3 = torch.stack(t1, dim=self.dim)
        
        return t3

# Initialize the model
model = UnbindConcatModel(dim=1)

# Generate input tensor
input_tensor = torch.randn(2, 3, 64, 64)  # Shape: (N, C, H, W)

# Pass the input tensor through the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
