import torch
import torch.nn as nn

class SplitModel(nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        self.fc = nn.Linear(10, 10)  # Simple linear layer for demonstration

    def forward(self, x):
        # Assume the input tensor is of shape (batch_size, 10)
        # Split the input into 2 parts: first 5 elements and last 5 elements
        sizes = [5, 5]  # Sizes for splitting
        splits = torch.split(x, sizes, dim=1)  # Split along the feature dimension
        
        # Use unique indices for the split parts
        output1 = splits[0]  # First part
        output2 = splits[1]  # Second part
        
        # For demonstration, we can concatenate the outputs
        output = torch.cat((output1, output2), dim=1)
        return output

# Initializing the model
model = SplitModel()

# Generate an input tensor
input_tensor = torch.randn(2, 10)  # Batch size of 2, feature size of 10

# Forward pass through the model
output = model(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Tensor:", output)
