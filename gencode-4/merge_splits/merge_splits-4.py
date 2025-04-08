import torch
import operator

class SplitModel(torch.nn.Module):
    def __init__(self, split_sizes):
        super().__init__()
        self.split_sizes = split_sizes
    
    def forward(self, input_tensor):
        # Split the input tensor along the first dimension
        splits = torch.split(input_tensor, self.split_sizes, dim=0)
        
        # Using unique indices to access the split outputs
        output1 = operator.getitem(splits, 0)  # First split
        output2 = operator.getitem(splits, 1)  # Second split
        output3 = operator.getitem(splits, 2)  # Third split
        
        # Return all outputs concatenated for demonstration
        return torch.cat((output1, output2, output3), dim=0)

# Define split sizes for the model
split_sizes = [2, 3, 5]  # Example sizes for the splits

# Initialize the model
model = SplitModel(split_sizes)

# Create an input tensor with a batch size that matches the sum of split sizes
input_tensor = torch.randn(10, 3, 64, 64)  # Batch size of 10

# Get the output from the model
output = model(input_tensor)

print(output.shape)  # Should output the shape of the concatenated results
