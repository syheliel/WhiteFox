import torch

class Model(torch.nn.Module):
    def __init__(self, dim=1):
        super(Model, self).__init__()
        self.dim = dim
    
    def forward(self, input_tensor):
        # Unbind the input tensor along the specified dimension
        t1 = torch.unbind(input_tensor, dim=self.dim)
        
        # For demonstration, we'll use the first tensor (i=0)
        t2 = t1[0]
        
        # Now concatenate the tensors along the specified dimension
        t3 = torch.cat((t2.unsqueeze(0),) + t1[1:], dim=self.dim)  # Stack t2 and the rest of the tensors from t1
        
        return t3

# Initializing the model
m = Model(dim=1)

# Inputs to the model
input_tensor = torch.randn(3, 4, 64)  # Shape: (B, C, H) = (3, 4, 64)

# Forward pass
output = m(input_tensor)

input_tensor = torch.randn(3, 4, 64)  # Random tensor with shape (3, 4, 64)
