import torch

# Model Definition
class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = torch.tanh(t1)  # Apply hyperbolic tangent activation
        return t2

# Initializing the model
model = TanhModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 input features
output = model(input_tensor)

print("Input Tensor:\n", input_tensor)
print("Output Tensor:\n", output)
