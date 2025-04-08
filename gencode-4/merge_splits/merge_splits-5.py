import torch
import operator

class SplitModel(torch.nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()

    def forward(self, input_tensor):
        # Example sizes for splitting
        sizes = (2, 3, 4)  # This will split the input tensor along the first dimension
        splits = torch.split(input_tensor, sizes, dim=0)

        # Unique getitem calls using unique non-negative integers
        output1 = splits[0]  # First split
        output2 = splits[1]  # Second split
        output3 = splits[2]  # Third split

        # Further operations can be performed on output1, output2, output3 if needed
        return output1, output2, output3

# Initializing the model
model = SplitModel()

# Generate an input tensor of shape (9, 3, 64, 64) to match the split sizes
input_tensor = torch.randn(9, 3, 64, 64)

# Forward pass through the model
outputs = model(input_tensor)

# Display the outputs
for i, output in enumerate(outputs):
    print(f"Output {i + 1} shape: {output.shape}")
