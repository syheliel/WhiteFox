import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions
        self.other = 2.0  # Constant to subtract from the linear output

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 - self.other  # Subtract 'other'
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initializing the model
model = Model()

# Generate an input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1, input dimension of 10
output = model(input_tensor)

print(output)
