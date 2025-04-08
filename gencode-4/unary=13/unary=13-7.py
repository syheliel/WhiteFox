import torch

# Model
class GatingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = torch.sigmoid(t1)  # Apply sigmoid function
        t3 = t1 * t2  # Multiply linear output by sigmoid output
        return t3

# Initializing the model
model = GatingModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 input features
output = model(input_tensor)

# Print the output
print(output)
