import torch

# Model
class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_A, input_tensor_B):
        # Permute input_tensor_A to swap the last two dimensions
        t1 = input_tensor_A.permute(0, 2, 1)  # Assuming input_tensor_A shape is (batch_size, seq_length_A, feature_A)
        # Permute input_tensor_B to swap the last two dimensions
        t2 = input_tensor_B.permute(0, 2, 1)  # Assuming input_tensor_B shape is (batch_size, seq_length_B, feature_B)
        # Perform batch matrix multiplication
        t3 = torch.bmm(t1, t2)  # Resulting shape will be (batch_size, feature_A, feature_B)
        return t3

# Initializing the model
model = MatrixMultiplicationModel()

# Inputs to the model
# Assuming input_tensor_A has shape (batch_size, seq_length_A, feature_A)
# Assuming input_tensor_B has shape (batch_size, seq_length_B, feature_B)
# Let’s say we have a batch size of 2, seq_length_A of 4, feature_A of 3, seq_length_B of 4, and feature_B of 5
input_tensor_A = torch.randn(2, 4, 3)  # Shape: (2, 4, 3)
input_tensor_B = torch.randn(2, 4, 5)  # Shape: (2, 4, 5)

# Getting the output from the model
output = model(input_tensor_A, input_tensor_B)
print(output.shape)  # Should print (2, 3, 5) as the result of the batch matrix multiplication
