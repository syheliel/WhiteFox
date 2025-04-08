import torch

# Model definition
class PermuteLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Defining a linear layer
        self.linear = torch.nn.Linear(16, 32)  # Example size, can be adjusted

    def forward(self, input_tensor):
        # Permute the input tensor; let's say input_tensor has shape (batch_size, channels, height, width)
        # We will permute it to swap the last two dimensions: (batch_size, channels, width, height)
        t1 = input_tensor.permute(0, 1, 3, 2)  # Swapping last two dimensions
        # Apply linear transformation on the permuted tensor
        # The linear layer expects input of shape (batch_size, input_features)
        # Here we will flatten the permuted tensor
        t1_flat = t1.view(t1.size(0), -1)  # Flatten the tensor (batch_size, channels * width * height)
        t2 = self.linear(t1_flat)  # Apply linear transformation
        return t2

# Initializing the model
model = PermuteLinearModel()

# Generating input tensor
batch_size = 4
channels = 3
height = 8
width = 8
input_tensor = torch.randn(batch_size, channels, height, width)

# Forward pass through the model
output = model(input_tensor)

# Displaying the output shape
print("Output shape:", output.shape)
