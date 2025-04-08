import torch
import operator

class SplitModel(torch.nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)  # Example linear layer

    def forward(self, input_tensor):
        # Assume sizes is a list of integers that sum to the first dimension of input_tensor
        sizes = [3, 3, 4]  # Example sizes for splitting
        splits = torch.split(input_tensor, sizes, dim=0)
        
        # Using unique indices for getitem calls
        output1 = operator.getitem(splits, 0)  # First split
        output2 = operator.getitem(splits, 1)  # Second split
        output3 = operator.getitem(splits, 2)  # Third split

        # You can perform additional operations on outputs, if needed
        return output1, output2, output3

# Initializing the model
model = SplitModel()

# Generate an input tensor
input_tensor = torch.randn(10, 10)  # Batching 10 samples with 10 features each
outputs = model(input_tensor)

# Outputs
output1, output2, output3 = outputs
print("Output 1:", output1)
print("Output 2:", output2)
print("Output 3:", output3)
