import torch

# Model definition
class LinearPermuteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Defining a linear layer with input features = 16 and output features = 32
        self.linear = torch.nn.Linear(16, 32)

    def forward(self, x):
        t1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        # Permuting the last two dimensions (32, 4) -> (4, 32)
        t2 = t1.permute(0, 2, 1)  # Assuming input x has shape (batch_size, 4, 16)
        return t2

# Initializing the model
model = LinearPermuteModel()

# Inputs to the model
# Creating a random input tensor with shape (batch_size=1, seq_length=4, features=16)
input_tensor = torch.randn(1, 4, 16)

# Forward pass through the model
output = model(input_tensor)

# Print the output shape for verification
print("Output shape:", output.shape)
