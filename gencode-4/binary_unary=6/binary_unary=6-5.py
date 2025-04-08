import torch
import torch.nn as nn

class LinearReLUModel(nn.Module):
    def __init__(self):
        super(LinearReLUModel, self).__init__()
        self.linear = nn.Linear(10, 5)  # Linear layer with input size 10 and output size 5
        self.other = torch.tensor(1.0)   # Subtracting 1.0 from the output of the linear transformation

    def forward(self, x):
        t1 = self.linear(x)              # Apply linear transformation
        t2 = t1 - self.other             # Subtract 'other' (1.0 in this case)
        t3 = torch.relu(t2)              # Apply ReLU activation
        return t3

# Initializing the model
model = LinearReLUModel()

# Creating an input tensor for the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10

# Forward pass through the model
output = model(input_tensor)

# Output the result
print("Output of the model:", output)
