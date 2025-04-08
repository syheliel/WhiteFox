import torch
import torch.nn as nn
import operator

class SplitModel(nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        self.fc = nn.Linear(10, 20)  # A simple linear layer for demonstration

    def forward(self, x):
        # Apply a linear transformation
        x = self.fc(x)
        
        # Split the tensor into 4 parts of sizes 5, 5, 5, and 5
        splits = torch.split(x, (5, 5, 5, 5), dim=1)
        
        # Unique getitem calls for the output of the split
        output1 = operator.getitem(splits, 0)  # First split
        output2 = operator.getitem(splits, 1)  # Second split
        output3 = operator.getitem(splits, 2)  # Third split
        output4 = operator.getitem(splits, 3)  # Fourth split
        
        # Perform some operations with the outputs (optional)
        result = output1 + output2 + output3 + output4  # Just an example operation
        return result

# Initializing the model
model = SplitModel()

# Generating an input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input dimension of 10

# Passing the input tensor through the model
output = model(input_tensor)

print(output)
