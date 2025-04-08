import torch

# Define the model
class LinearPermuteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with input features and output features
        self.linear = torch.nn.Linear(128, 64)  # Example: from 128 input features to 64 output features

    def forward(self, x):
        # Apply the linear transformation
        t1 = self.linear(x)  # Apply linear transformation to the input tensor
        # Permute the output tensor (swap the last two dimensions)
        t2 = t1.permute(0, 2, 1)  # Assuming t1 is of shape (batch_size, seq_length, features)
        return t2

# Initializing the model
model = LinearPermuteModel()

# Create an input tensor with the shape (batch_size, seq_length, input_features)
# Here we assume batch_size=1, seq_length=4, input_features=128
input_tensor = torch.randn(1, 4, 128)

# Get the output from the model
output_tensor = model(input_tensor)

print("Output shape:", output_tensor.shape)  # Expected output shape: (1, 64, 4)
