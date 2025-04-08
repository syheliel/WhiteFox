import torch

class ConcatenationModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input_tensors):
        # Concatenate input tensors along dimension 1
        t1 = torch.cat(input_tensors, dim=1)

        # Slicing the tensor; using a large number for illustration
        t2 = t1[:, 0:9223372036854775807]

        # Slicing the tensor from index 0 to size
        t3 = t2[:, 0:self.size]

        # Concatenate the original tensor and the sliced tensor along dimension 1
        t4 = torch.cat([t1, t3], dim=1)

        return t4

# Initialize the model with a specific size
size = 10  # Example size for slicing
model = ConcatenationModel(size)

# Generate input tensors for the model
input_tensor1 = torch.randn(1, 3, 64, 64)  # Example input tensor 1
input_tensor2 = torch.randn(1, 3, 64, 64)  # Example input tensor 2
input_tensors = [input_tensor1, input_tensor2]  # List of input tensors

# Forward pass through the model
output = model(input_tensors)

# Print the output shape
print(output.shape)
