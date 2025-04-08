import torch

# Model definition
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features

    def forward(self, x):
        t1 = self.linear(x)           # Apply a linear transformation to the input tensor
        t2 = t1 + 3                   # Add 3 to the output of the linear transformation
        t3 = torch.clamp_min(t2, 0)   # Clamp the output of the addition operation to a minimum of 0
        t4 = torch.clamp_max(t3, 6)   # Clamp the output of the previous operation to a maximum of 6
        t5 = t4 / 6                   # Divide the output of the previous operation by 6
        return t5

# Initializing the model
model = LinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 features
output = model(input_tensor)

# Print the output
print("Output of the model:", output)
