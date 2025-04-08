import torch

class Model(torch.nn.Module):
    def __init__(self, dim=1, i=0):
        super().__init__()
        self.dim = dim
        self.i = i

    def forward(self, input_tensor):
        # Unbind the input tensor along a specified dimension
        t1 = torch.unbind(input_tensor, dim=self.dim)
        
        # Get the i-th tensor from the unbound tensors
        t2 = t1[self.i]
        
        # Stack the tensor along the specified dimension
        t3 = torch.stack(t1, dim=self.dim)
        
        return t3

# Initializing the model
m = Model(dim=1, i=0)

# Inputs to the model
input_tensor = torch.randn(3, 4, 64)  # A tensor with shape (3, 4, 64)
output = m(input_tensor)

print(output.shape)  # This will show the shape of the output tensor
