import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define two linear layers for demonstration
        self.linear1 = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(10, 5)

    def forward(self, tensor1, tensor2):
        # Concatenate the two tensors along the last dimension
        t1 = torch.cat([tensor1, tensor2], dim=-1)
        
        # Reshape the concatenated tensor to have 5 rows
        t2 = t1.view(-1, 5)
        
        # Apply ReLU activation function to the reshaped tensor
        t3 = torch.relu(t2)
        
        return t3

# Initializing the model
model = Model()

# Generate input tensors
tensor1 = torch.randn(2, 10)  # Shape (2, 10)
tensor2 = torch.randn(2, 10)  # Shape (2, 10)

# Forward pass
output = model(tensor1, tensor2)

print("Output shape:", output.shape)
print("Output tensor:", output)
