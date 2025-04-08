import torch

class PermuteAndMultiplyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_A, input_tensor_B):
        # Permute the last two dimensions of input_tensor_A and input_tensor_B
        t1 = input_tensor_A.permute(0, 2, 1)  # Assuming input_tensor_A has shape (batch_size, n, m)
        t2 = input_tensor_B.permute(0, 2, 1)  # Assuming input_tensor_B has shape (batch_size, p, q)

        # Perform batch matrix multiplication
        t3 = torch.bmm(t1, t2)  # Now t3 will have shape (batch_size, m, q)

        return t3

# Initializing the model
model = PermuteAndMultiplyModel()

# Inputs to the model
# Assuming batch size of 1 for example, and the shapes of the input tensors
input_tensor_A = torch.randn(1, 4, 3)  # Shape (batch_size, n, m)
input_tensor_B = torch.randn(1, 3, 5)  # Shape (batch_size, p, q)

# Forward pass
output = model(input_tensor_A, input_tensor_B)

# Output shape
print("Output shape:", output.shape)  # Should be (1, 3, 5)
