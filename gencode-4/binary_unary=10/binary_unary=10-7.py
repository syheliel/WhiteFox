import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
        self.other_tensor = torch.randn(1, 5)  # This will be the tensor added to the output of the linear layer

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = t1 + self.other_tensor  # Add another tensor
        t3 = nn.functional.relu(t2)  # Apply ReLU activation
        return t3

# Initializing the model
model = CustomModel()

# Input to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 features

# Getting the output of the model
output = model(input_tensor)
print(output)
