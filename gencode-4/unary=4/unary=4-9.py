import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with 10 input features and 5 output features
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, input_tensor):
        l1 = self.linear(input_tensor)  # Apply pointwise linear transformation
        l2 = l1 * 0.5                    # Multiply by 0.5
        l3 = l1 * 0.7071067811865476     # Multiply by 0.7071067811865476
        l4 = torch.erf(l3)               # Apply the error function
        l5 = l4 + 1                       # Add 1 to the output of the error function
        l6 = l2 * l5                     # Multiply l2 by l5
        return l6

# Initializing the model
model = LinearModel()

# Generating input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 input features
output = model(input_tensor)

# Displaying the output shape
print("Output shape:", output.shape)
