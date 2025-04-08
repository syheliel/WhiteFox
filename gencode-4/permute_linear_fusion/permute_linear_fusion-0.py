import torch
import torch.nn as nn

class PermuteLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Assuming input tensor has shape (batch_size, channels, height, width)
        # We will permute to (batch_size, height, width, channels) for the linear layer
        self.linear = nn.Linear(8 * 64, 16)  # Example: input features = 8*64, output features = 16

    def forward(self, x):
        # Permute the input tensor to swap the last two dimensions
        t1 = x.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        t1 = t1.contiguous()  # Ensure the tensor is contiguous after permutation
        
        # Flatten the last two dimensions for the linear layer
        t1 = t1.view(t1.size(0), -1)  # (batch_size, height*width*channels)
        
        # Apply the linear transformation
        t2 = self.linear(t1)
        return t2

# Initializing the model
model = PermuteLinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor with shape (batch_size, channels, height, width)
output = model(input_tensor)

# Output shape
print("Output shape:", output.shape)
