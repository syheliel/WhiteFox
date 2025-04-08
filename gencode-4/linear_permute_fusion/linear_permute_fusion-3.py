import torch

class LinearPermuteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with input features and output features
        self.linear = torch.nn.Linear(16, 32)  # Change input and output dimensions as needed

    def forward(self, x):
        # Apply linear transformation to the input tensor
        t1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        # Permute the output tensor (e.g., swap last two dimensions)
        t2 = t1.permute(0, 2, 1)  # Assuming t1 is of shape (batch_size, seq_len, features)
        return t2

# Initializing the model
model = LinearPermuteModel()

# Generating input tensor
input_tensor = torch.randn(4, 4, 16)  # Batch size of 4, sequence length of 4, and feature size of 16
output = model(input_tensor)

print("Output shape:", output.shape)
