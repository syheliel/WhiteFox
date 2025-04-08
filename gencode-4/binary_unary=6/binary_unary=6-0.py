import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5)  # Linear layer with input size 10 and output size 5
        self.other = torch.tensor(0.5)   # The value to subtract

    def forward(self, x):
        t1 = self.linear(x)              # Apply linear transformation
        t2 = t1 - self.other             # Subtract 'other' from the output of linear transformation
        t3 = torch.relu(t2)              # Apply ReLU activation function
        return t3

# Initializing the model
model = SimpleModel()

# Input tensor for the model
input_tensor = torch.randn(3, 10)  # Batch size of 3 and input size of 10
output_tensor = model(input_tensor)

# Display the output
print(output_tensor)
