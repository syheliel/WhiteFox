import torch
import torch.nn as nn

class SplitModel(nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        # A simple linear layer to transform input
        self.linear = nn.Linear(10, 20)  # Input size 10, output size 20

    def forward(self, x):
        # Apply the linear transformation
        transformed = self.linear(x)
        
        # Split the tensor into parts of sizes [5, 5, 10]
        splits = torch.split(transformed, [5, 5, 10], dim=1)
        
        # Use each split with a unique non-negative index
        output1 = splits[0]  # First part
        output2 = splits[1]  # Second part
        output3 = splits[2]  # Third part

        # Combine outputs in some way (e.g., concatenating)
        combined = torch.cat((output1, output2, output3), dim=1)
        return combined

# Initialize the model
model = SplitModel()

# Generate a random input tensor of shape (batch_size, input_size)
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10

# Forward pass through the model
output = model(input_tensor)

print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output.shape)
