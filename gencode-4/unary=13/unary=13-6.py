import torch

# Model
class GatedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)  # Linear transformation

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = torch.sigmoid(t1)  # Apply sigmoid function
        t3 = t1 * t2  # Multiply the output of the linear transformation by the output of the sigmoid function
        return t3

# Initializing the model
gated_model = GatedModel()

# Inputs to the model
x_input = torch.randn(5, 10)  # Batch size of 5, input features of size 10
output = gated_model(x_input)

print(output)
