import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Example layers can be added if needed, but for this model, we're focusing on the specified operations.

    def forward(self, input_tensor_A, input_tensor_B):
        # Permute the input tensors
        t1 = input_tensor_A.permute(0, 2, 1)  # Swap the last two dimensions of input_tensor_A
        t2 = input_tensor_B.permute(0, 2, 1)  # Swap the last two dimensions of input_tensor_B
        
        # Perform batch matrix multiplication
        t3 = torch.bmm(t1, t2)  # Batch matrix multiplication
        return t3

# Initialize the model
model = CustomModel()

# Generate input tensors
# Assuming input_tensor_A has shape (batch_size, num_rows_A, num_cols_A)
# and input_tensor_B has shape (batch_size, num_rows_B, num_cols_B)
# For this example, let's say both tensors have a batch size of 2 and compatible dimensions
input_tensor_A = torch.randn(2, 4, 3)  # Shape: (batch_size=2, num_rows_A=4, num_cols_A=3)
input_tensor_B = torch.randn(2, 5, 4)  # Shape: (batch_size=2, num_rows_B=5, num_cols_B=4)

# Forward pass through the model
output = model(input_tensor_A, input_tensor_B)

# Display the output shape
print(f"Output shape: {output.shape}")
