import torch
import torch.nn as nn
import operator

class SplitModel(nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        self.fc = nn.Linear(20, 30)  # Fully connected layer

    def forward(self, x):
        # Apply the fully connected layer
        x = self.fc(x)
        
        # Split the tensor into parts
        sizes = [10, 10, 10]  # Example sizes for splitting
        splits = torch.split(x, sizes, dim=1)  # Split along the second dimension
        
        # Unique getitem calls
        output1 = operator.getitem(splits, 0)  # First split
        output2 = operator.getitem(splits, 1)  # Second split
        output3 = operator.getitem(splits, 2)  # Third split
        
        return output1, output2, output3

# Initialize the model
model = SplitModel()

# Generate an input tensor
input_tensor = torch.randn(5, 20)  # Batch size of 5 and 20 features
outputs = model(input_tensor)

input_tensor = torch.randn(5, 20)  # A batch of 5 samples, each with 20 features
