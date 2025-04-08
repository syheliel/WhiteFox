import torch
import torch.nn as nn

class SplitModel(nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        self.fc = nn.Linear(16, 32)  # A simple linear layer

    def forward(self, x):
        # Apply the linear transformation
        x = self.fc(x)

        # Split the output tensor into specified sizes
        sizes = (8, 8, 16)  # Example sizes for splitting
        splits = torch.split(x, sizes, dim=0)

        # Unique non-negative indices for each split
        output1 = splits[0]  # First split (first 8 elements)
        output2 = splits[1]  # Second split (next 8 elements)
        output3 = splits[2]  # Third split (remaining 16 elements)

        # Combine outputs (just for demonstration)
        combined_output = output1 + output2 + output3
        return combined_output

# Initialize the model
model = SplitModel()

# Generate an input tensor for the model
input_tensor = torch.randn(32, 16)  # Batch size of 32 and 16 features
output = model(input_tensor)

# Print output shape
print(f"Output shape: {output.shape}")
