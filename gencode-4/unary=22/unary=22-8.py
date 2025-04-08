import torch

# Model definition
class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
    
    def forward(self, x):
        t1 = self.linear(x)          # Apply a linear transformation
        t2 = torch.tanh(t1)         # Apply the hyperbolic tangent function
        return t2

# Initializing the model
model = TanhModel()

# Input to the model
input_tensor = torch.randn(3, 10)  # Batch of 3, with 10 features

# Getting the output from the model
output = model(input_tensor)

# Output the shape of the model's output for confirmation
print("Output shape:", output.shape)
