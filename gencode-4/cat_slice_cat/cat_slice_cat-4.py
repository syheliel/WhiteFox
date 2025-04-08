import torch

# Model
class ConcatenateModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input_tensors):
        # Concatenate input tensors along dimension 1
        t1 = torch.cat(input_tensors, dim=1)

        # Slice the tensor along dimension 1 from index 0 to a very large number
        t2 = t1[:, 0:9223372036854775807]  # Note: This will not actually slice anything in practice

        # Slice the tensor along dimension 1 from index 0 to specified size
        t3 = t2[:, 0:self.size]

        # Concatenate the original tensor and the sliced tensor along dimension 1
        t4 = torch.cat([t1, t3], dim=1)

        return t4

# Example size for slicing
size = 16

# Initializing the model
model = ConcatenateModel(size)

# Generating input tensors for the model
# Assuming we want to concatenate two input tensors of shape (1, 3, 64, 64)
input_tensor1 = torch.randn(1, 3, 64, 64)  # First input tensor
input_tensor2 = torch.randn(1, 5, 64, 64)  # Second input tensor (different channel size)

# Input list for concatenation
input_tensors = [input_tensor1, input_tensor2]

# Forward pass
output = model(input_tensors)

# Display the output shape
print(output.shape)  # Should reflect the concatenated shape
