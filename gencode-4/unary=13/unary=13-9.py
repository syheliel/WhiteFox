import torch

# Model
class GatedLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # A linear layer with input size 10 and output size 5

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = torch.sigmoid(t1)  # Apply the sigmoid function to the output of the linear transformation
        t3 = t1 * t2  # Multiply the output of the linear transformation by the output of the sigmoid function
        return t3

# Initializing the model
model = GatedLinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # An input tensor with batch size 1 and 10 features

# Forward pass
output = model(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("Output Tensor:")
print(output)
