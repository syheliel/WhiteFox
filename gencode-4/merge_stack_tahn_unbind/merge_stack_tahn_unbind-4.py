import torch

class CustomModel(torch.nn.Module):
    def __init__(self, split_size):
        super().__init__()
        self.split_size = split_size

    def forward(self, x):
        # Split the input tensor into chunks along the specified dimension
        t1 = torch.split(x, self.split_size, dim=1)  # Split along the channel dimension (dim=1)
        
        # Stack the chunks along a new dimension
        t2 = torch.stack(t1, dim=2)  # Stack along a new dimension (2)

        # Apply the hyperbolic tangent function
        t3 = torch.tanh(t2)
        
        return t3

# Initialize the model with a specified split size
split_size = 2  # Example: Split the input tensor along the channel dimension into chunks of 2
model = CustomModel(split_size)

# Generate input tensor
input_tensor = torch.randn(1, 4, 64, 64)  # Batch size of 1, 4 channels, 64x64 spatial dimensions

# Forward pass through the model
output = model(input_tensor)
