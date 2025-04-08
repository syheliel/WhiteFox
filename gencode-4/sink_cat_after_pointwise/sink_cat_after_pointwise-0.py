import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(20, 10)  # First linear layer
        self.linear2 = torch.nn.Linear(20, 10)  # Second linear layer

    def forward(self, tensor1, tensor2):
        # Concatenate tensors along the last dimension
        t1 = torch.cat([tensor1, tensor2], dim=-1)
        
        # Reshape the concatenated tensor
        t2 = t1.view(-1, 20)  # Reshaping to have 20 features per input
        
        # Apply ReLU activation function
        t3 = torch.relu(t2)  # Using ReLU as the pointwise unary operation
        
        return t3

# Initializing the model
model = Model()

# Generating input tensors
tensor1 = torch.randn(5, 10)  # A batch of 5 samples, each with 10 features
tensor2 = torch.randn(5, 10)  # Another batch of 5 samples, each with 10 features

# Forward pass through the model
output = model(tensor1, tensor2)

# Print output shape
print("Output shape:", output.shape)
