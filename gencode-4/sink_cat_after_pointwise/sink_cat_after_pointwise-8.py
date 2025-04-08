import torch

class ConcatenationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, tensor1, tensor2):
        # Concatenate tensors along the last dimension
        t1 = torch.cat([tensor1, tensor2], dim=-1)
        
        # Reshape the concatenated tensor
        t2 = t1.view(t1.size(0), -1)  # Flatten all dimensions except the batch size
        
        # Apply ReLU activation function
        t3 = torch.relu(t2)
        
        return t3

# Initializing the model
model = ConcatenationModel()

# Generating input tensors
tensor1 = torch.randn(1, 3, 64)  # Example tensor with shape (1, 3, 64)
tensor2 = torch.randn(1, 3, 64)  # Another tensor with the same shape

# Getting the output from the model
output = model(tensor1, tensor2)

print(output.shape)  # Print the shape of the output tensor
