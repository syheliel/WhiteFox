import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size  # Store the size for slicing

    def forward(self, input_tensors):
        t1 = torch.cat(input_tensors, dim=1)  # Concatenate input tensors along dimension 1
        t2 = t1[:, 0:9223372036854775807]  # Slice the tensor along dimension 1
        t3 = t2[:, 0:self.size]  # Slice the tensor along dimension 1 from index 0 to size
        t4 = torch.cat([t1, t3], dim=1)  # Concatenate the original tensor and the sliced tensor along dimension 1
        return t4

# Initializing the model with a specified size
size = 10  # Example size
model = Model(size)

# Generating input tensors for the model
input_tensor1 = torch.randn(1, 3, 64, 64)  # Example tensor with shape (1, 3, 64, 64)
input_tensor2 = torch.randn(1, 5, 64, 64)  # Another tensor with shape (1, 5, 64, 64)
input_tensors = [input_tensor1, input_tensor2]  # List of input tensors

# Forward pass through the model
output = model(input_tensors)

# Display the output shape
print(output.shape)
