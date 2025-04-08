import torch

class PermuteLinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PermuteLinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Assuming x has shape (batch_size, channels, height, width)
        # Permuting to (batch_size, height, width, channels)
        t1 = x.permute(0, 2, 3, 1)  # Swap the last two dimensions
        # Prepare for linear transformation
        # Reshape t1 to (batch_size * height * width, channels) for linear layer
        t1_flat = t1.view(-1, t1.size(-1))
        # Apply linear transformation
        t2 = self.linear(t1_flat)
        # Reshape back to (batch_size, height, width, output_dim)
        return t2.view(x.size(0), x.size(2), x.size(3), -1)

# Initializing the model
input_dim = 16  # Number of input features (channels)
output_dim = 32  # Number of output features
model = PermuteLinearModel(input_dim, output_dim)

# Inputs to the model
# Create an input tensor of shape (batch_size, channels, height, width)
x1 = torch.randn(1, input_dim, 64, 64)  # Example input tensor with batch size 1

# Forward pass through the model
output = model(x1)

# Display the output shape
print(output.shape)  # Should print: (1, 64, 64, 32)
