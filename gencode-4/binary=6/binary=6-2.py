import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions
    
    def forward(self, x, other):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = t1 - other      # Subtract 'other' from the output of the linear transformation
        return t2

# Initializing the model
model = LinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input dimension of 10
other = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])  # A tensor to subtract from the output

# Forward pass
output = model(input_tensor, other)

print("Input Tensor:\n", input_tensor)
print("Output Tensor:\n", output)
