import torch

class CustomModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size  # Store the size for slicing

    def forward(self, *input_tensors):
        t1 = torch.cat(input_tensors, dim=1)  # Concatenate input tensors along dimension 1
        t2 = t1[:, 0:9223372036854775807]  # Slice the tensor along dimension 1 from index 0 to a large number
        t3 = t2[:, 0:self.size]  # Slice the tensor along dimension 1 from index 0 to specified size
        t4 = torch.cat([t1, t3], dim=1)  # Concatenate the original tensor and the sliced tensor along dimension 1
        return t4

# Initialize the model
size = 10  # Define the size for the slicing operation
model = CustomModel(size)

# Generate input tensors
input_tensor1 = torch.randn(1, 5, 64, 64)  # Example tensor of shape (1, 5, 64, 64)
input_tensor2 = torch.randn(1, 3, 64, 64)  # Example tensor of shape (1, 3, 64, 64)

# Forward pass with the generated input tensors
output = model(input_tensor1, input_tensor2)

# Print the output shape
print("Output shape:", output.shape)
