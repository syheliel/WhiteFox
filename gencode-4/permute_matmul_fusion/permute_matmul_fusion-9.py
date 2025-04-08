import torch

class MatrixModel(torch.nn.Module):
    def __init__(self):
        super(MatrixModel, self).__init__()
    
    def forward(self, input_tensor_A, input_tensor_B):
        # Permute the input tensors to swap the last two dimensions
        t1 = input_tensor_A.permute(0, 2, 1)  # Assuming input_tensor_A has shape (batch_size, seq_length, features)
        t2 = input_tensor_B.permute(0, 2, 1)  # Assuming input_tensor_B has shape (batch_size, seq_length, features)

        # Perform batch matrix multiplication
        t3 = torch.bmm(t1, t2)  # Resulting shape will be (batch_size, features, features)
        return t3

# Initialize the model
model = MatrixModel()

# Generate input tensors
# Assume we have a batch size of 2, sequence length of 4, and features of 3
input_tensor_A = torch.randn(2, 4, 3)  # Shape: (batch_size, seq_length, features)
input_tensor_B = torch.randn(2, 4, 3)  # Shape: (batch_size, seq_length, features)

# Get the output by passing the input tensors through the model
output = model(input_tensor_A, input_tensor_B)

# Output shape
print("Output shape:", output.shape)  # Should be (batch_size, features, features)
