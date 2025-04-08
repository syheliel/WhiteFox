import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
        self.other = 0.5  # The value to be subtracted from the linear output

    def forward(self, input_tensor):
        t1 = self.linear(input_tensor)  # Apply linear transformation
        t2 = t1 - self.other  # Subtract 'other'
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initializing the model
model = Model()

# Generating the input tensor
input_tensor = torch.randn(1, 10)  # Example input tensor with 1 sample and 10 features

# Running the model with the input tensor
output = model(input_tensor)

# Output the result
print(output)
