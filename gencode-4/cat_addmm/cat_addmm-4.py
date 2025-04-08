import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)  # First linear layer
        self.linear2 = torch.nn.Linear(5, 5)   # Second linear layer
        self.concat_dim = 1  # Dimension along which to concatenate

    def forward(self, input_tensor):
        # Perform matrix multiplication of the first linear layer and add to input
        t1 = torch.addmm(input_tensor, self.linear1.weight, self.linear1.bias.unsqueeze(0))
        
        # Perform matrix multiplication of the second linear layer
        t1 = torch.addmm(t1, self.linear2.weight, self.linear2.bias.unsqueeze(0))
        
        # Concatenate along the specified dimension
        t2 = torch.cat([t1], dim=self.concat_dim)
        
        return t2

# Initialize the model
model = Model()

# Generate an input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and feature size of 10

# Forward pass through the model
output = model(input_tensor)

# Display the output
print("Output shape:", output.shape)
