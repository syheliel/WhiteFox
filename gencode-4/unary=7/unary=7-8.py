import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features

    def forward(self, x):
        l1 = self.linear(x)  # Apply linear transformation
        l2 = l1 + 3          # Add 3
        l3 = torch.clamp(l2, min=0)  # Clamp to min 0
        l4 = torch.clamp(l3, max=6)  # Clamp to max 6
        l5 = l1 * l4         # Multiply by output of the clamp operation
        l6 = l5 / 6          # Divide by 6
        return l6

# Initializing the model
model = Model()

# Generating the input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 input features
output = model(input_tensor)

# Display the output
print(output)
