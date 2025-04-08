import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with input features of 10 and output features of 5
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x):
        l1 = self.linear(x)  # Apply pointwise linear transformation to the input tensor
        l2 = l1 * 0.5  # Multiply the output of the linear transformation by 0.5
        l3 = l1 * 0.7071067811865476  # Multiply the output of the linear transformation by 0.7071067811865476
        l4 = torch.erf(l3)  # Apply the error function to the output of the linear transformation
        l5 = l4 + 1  # Add 1 to the output of the error function
        l6 = l2 * l5  # Multiply the output of the linear transformation by the output of the error function
        return l6

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 input features
output = model(input_tensor)

print("Output of the model:", output)
