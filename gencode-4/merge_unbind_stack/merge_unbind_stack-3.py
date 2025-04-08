import torch

class Model(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # Unbind the input tensor along the specified dimension
        t1 = torch.unbind(input_tensor, dim=self.dim)
        # Get the i-th tensor from the unbound tensors (let's take the first one as an example)
        t2 = t1[0]
        # Stack the tensors along the specified dimension
        t3 = torch.stack(t1, dim=self.dim)  # This can also be replaced with torch.cat(t1, dim=self.dim)
        return t3

# Initializing the model
model = Model(dim=1)

# Inputs to the model
input_tensor = torch.randn(4, 3, 64, 64)  # Example input tensor of shape (4, 3, 64, 64)
output = model(input_tensor)

print("Output shape:", output.shape)
