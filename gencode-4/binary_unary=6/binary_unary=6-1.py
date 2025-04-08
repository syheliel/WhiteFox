import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear layer with 10 input features and 5 output features
        self.other = 1.0  # The constant value to subtract

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 - self.other  # Subtract 'other' from the linear output
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Random input tensor with batch size of 1 and 10 features
output = model(input_tensor)

print("Output of the model:", output)
