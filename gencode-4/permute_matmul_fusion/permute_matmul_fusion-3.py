import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor_A, input_tensor_B):
        # Permute the input tensors to swap the last two dimensions
        t1 = input_tensor_A.permute(0, 2, 1)  # Assuming input_tensor_A has shape (batch_size, seq_len_A, features)
        t2 = input_tensor_B.permute(0, 2, 1)  # Assuming input_tensor_B has shape (batch_size, seq_len_B, features)
        
        # Perform batch matrix multiplication
        t3 = torch.bmm(t1, t2)  # Resulting shape will be (batch_size, seq_len_A, seq_len_B)
        return t3

# Initializing the model
model = Model()

# Generate input tensors
# Assuming both input tensors have shape (batch_size, seq_len, features)
input_tensor_A = torch.randn(2, 4, 3)  # Example: batch size = 2, seq_len_A = 4, features = 3
input_tensor_B = torch.randn(2, 5, 3)  # Example: batch size = 2, seq_len_B = 5, features = 3

# Get the model output
output = model(input_tensor_A, input_tensor_B)

# Print the shapes of the output
print("Output shape:", output.shape)  # Should be (batch_size, seq_len_A, seq_len_B)
