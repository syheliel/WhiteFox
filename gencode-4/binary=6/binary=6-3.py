import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions
        self.other = 2.0  # Scalar value to subtract

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation to the input tensor
        t2 = t1 - self.other  # Subtract 'other' from the output of the linear transformation
        return t2

# Initializing the model
model = LinearModel()

# Generating an input tensor for the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input dimension of 10
output = model(input_tensor)

# Display the output
print(output)
