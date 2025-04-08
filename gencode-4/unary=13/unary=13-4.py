import torch
import torch.nn as nn

class GatedModel(nn.Module):
    def __init__(self):
        super(GatedModel, self).__init__()
        self.linear = nn.Linear(10, 5)  # Example input size of 10 and output size of 5

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = torch.sigmoid(t1)  # Apply sigmoid to the output of the linear transformation
        t3 = t1 * t2  # Multiply the output of the linear transformation by the output of the sigmoid
        return t3

# Initializing the model
model = GatedModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input feature size of 10
output = model(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("\nOutput Tensor:")
print(output)
