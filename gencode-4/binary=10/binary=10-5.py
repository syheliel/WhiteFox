import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear layer from 10 input features to 5 output features
        self.other = torch.randn(1, 5)  # Another tensor to add, initialized with random values

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 + self.other  # Add the other tensor to the output of the linear transformation
        return t2

# Initializing the model
model = LinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Random input tensor of shape (1, 10)
output = model(input_tensor)

print("Output Tensor:", output)
