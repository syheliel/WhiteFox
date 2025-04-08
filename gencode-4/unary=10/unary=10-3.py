import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features

    def forward(self, x):
        t1 = self.linear(x)          # Apply linear transformation
        t2 = t1 + 3                  # Add 3
        t3 = torch.clamp_min(t2, 0)  # Clamp minimum to 0
        t4 = torch.clamp_max(t3, 6)  # Clamp maximum to 6
        t5 = t4 / 6                  # Divide by 6
        return t5

# Initialize the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1, with 10 features

# Get the output of the model
output = model(input_tensor)

# Print the output
print("Output of the model:", output)
