import torch
import torch.nn as nn

class SplitModel(nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        self.fc = nn.Linear(20, 20)  # A fully connected layer for demonstration

    def forward(self, x):
        # Pass the input through the fully connected layer
        x = self.fc(x)
        
        # Split the tensor into 5 chunks of equal size
        splits = torch.split(x, 4, dim=1)  # Assume input has feature size 20
        
        # Use unique non-negative integers for indexing
        output1 = splits[0]  # First chunk
        output2 = splits[1]  # Second chunk
        output3 = splits[2]  # Third chunk
        output4 = splits[3]  # Fourth chunk
        
        # Combine outputs in some way, for example
        combined_output = output1 + output2 + output3 + output4
        return combined_output

# Initializing the model
model = SplitModel()

# Creating an input tensor of shape (batch_size, features)
# Here, we assume batch size is 1 and features are 20
input_tensor = torch.randn(1, 20)

# Forward pass through the model
output_tensor = model(input_tensor)

print("Output Tensor:", output_tensor)
