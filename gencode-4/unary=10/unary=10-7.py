import torch

# Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions

    def forward(self, x1):
        t1 = self.linear(x1)          # Apply a linear transformation to the input tensor
        t2 = t1 + 3                   # Add 3 to the output of the linear transformation
        t3 = torch.clamp_min(t2, 0)   # Clamp the output of the addition operation to a minimum of 0
        t4 = torch.clamp_max(t3, 6)   # Clamp the output of the previous operation to a maximum of 6
        t5 = t4 / 6                    # Divide the output of the previous operation by 6
        return t5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)  # Input tensor of shape (1, 10)
output = m(x1)           # Get the model output

# Print the output
print(output)
