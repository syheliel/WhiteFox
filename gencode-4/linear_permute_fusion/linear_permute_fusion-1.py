import torch

# Model Definition
class LinearPermuteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the linear layer with appropriate input and output features
        self.linear = torch.nn.Linear(128, 64)  # Example: input features = 128, output features = 64
    
    def forward(self, x):
        t1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)  # Apply linear transformation
        t2 = t1.permute(0, 2, 1)  # Permute the last two dimensions (assuming input has at least 3 dimensions)
        return t2

# Initializing the model
model = LinearPermuteModel()

# Generating input tensor
# Example input tensor with shape (batch_size, sequence_length, input_features)
input_tensor = torch.randn(10, 8, 128)  # Batch size = 10, Sequence length = 8, Input features = 128

# Forward pass through the model
output_tensor = model(input_tensor)

print("Output tensor shape:", output_tensor.shape)
