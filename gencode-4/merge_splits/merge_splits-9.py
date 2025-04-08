import torch

class SplitModel(torch.nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        self.linear = torch.nn.Linear(10, 20)  # A simple linear layer

    def forward(self, x):
        # Apply linear transformation
        x = self.linear(x)
        
        # Split the output tensor into segments
        sizes = [5, 5, 10]  # Sizes for the split
        splits = torch.split(x, sizes, dim=1)  # Split along the feature dimension
        
        # Each output from split is indexed uniquely
        output1 = splits[0]  # First segment
        output2 = splits[1]  # Second segment
        output3 = splits[2]  # Third segment
        
        # Combine outputs for demonstration (optional)
        combined_output = output1 + output2 + output3
        return combined_output

# Initializing the model
model = SplitModel()

# Generating an input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1, 10 features
output = model(input_tensor)

# Displaying the output
print(output)
