import torch

# Model Definition
class ConcatenationModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size  # The size to which we slice the tensor
    
    def forward(self, input_tensors):
        # Concatenate input tensors along dimension 1
        t1 = torch.cat(input_tensors, dim=1)  
        
        # Slice the tensor along dimension 1 from index 0 to a very large number
        t2 = t1[:, 0:9223372036854775807]  
        
        # Slice the tensor from index 0 to 'size'
        t3 = t2[:, 0:self.size]  
        
        # Concatenate the original tensor and the sliced tensor along dimension 1
        t4 = torch.cat([t1, t3], dim=1)  
        
        return t4

# Example usage
size = 10  # Define the size for slicing
model = ConcatenationModel(size)

# Generating Input Tensors
input_tensor1 = torch.randn(1, 3, 64, 64)  # First input tensor with shape [1, 3, 64, 64]
input_tensor2 = torch.randn(1, 4, 64, 64)  # Second input tensor with shape [1, 4, 64, 64]
input_tensors = [input_tensor1, input_tensor2]  # List of input tensors

# Forward pass through the model
output = model(input_tensors)

print("Output shape:", output.shape)
