import torch

# Define the model
class ConcatenationModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input_tensors):
        # Concatenate input tensors along dimension 1
        t1 = torch.cat(input_tensors, dim=1)
        
        # Slice the tensor along dimension 1 from index 0 to a very large number
        t2 = t1[:, 0:9223372036854775807]
        
        # Slice the tensor along dimension 1 from index 0 to specified size
        t3 = t2[:, 0:self.size]
        
        # Concatenate the original tensor and the sliced tensor along dimension 1
        t4 = torch.cat([t1, t3], dim=1)
        
        return t4

# Initialize the model
size = 10  # Define size for slicing
model = ConcatenationModel(size)

# Generate input tensors
input_tensor1 = torch.randn(1, 5, 64, 64)  # Example input tensor 1
input_tensor2 = torch.randn(1, 3, 64, 64)  # Example input tensor 2
input_tensors = [input_tensor1, input_tensor2]

# Forward pass
output = model(input_tensors)

# Print output shape
print(output.shape)

input_tensor1 = torch.randn(1, 5, 64, 64)  # Shape: (1, 5, 64, 64)
input_tensor2 = torch.randn(1, 3, 64, 64)  # Shape: (1, 3, 64, 64)
