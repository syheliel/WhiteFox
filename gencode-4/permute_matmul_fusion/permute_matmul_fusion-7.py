import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_A, input_tensor_B):
        # Permute the last two dimensions of input_tensor_A
        t1 = input_tensor_A.permute(0, 2, 1)  # Assuming input_tensor_A has shape (batch_size, seq_length_A, features)
        # Permute the last two dimensions of input_tensor_B
        t2 = input_tensor_B.permute(0, 2, 1)  # Assuming input_tensor_B has shape (batch_size, seq_length_B, features)
        # Batch matrix multiplication
        t3 = torch.bmm(t1, t2)  # Resulting shape will be (batch_size, features, features)
        return t3

# Initializing the model
model = CustomModel()

# Inputs to the model
# Define input tensor A with shape (batch_size, seq_length_A, features)
input_tensor_A = torch.randn(2, 4, 3)  # Example: batch size of 2, sequence length of 4, features of 3
# Define input tensor B with shape (batch_size, seq_length_B, features)
input_tensor_B = torch.randn(2, 5, 3)  # Example: batch size of 2, sequence length of 5, features of 3

# Forward pass
output = model(input_tensor_A, input_tensor_B)

# Print the output shape
print(output.shape)  # Expected output shape: (batch_size, features, features)
