import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions

    def forward(self, x):
        t1 = self.linear(x)          # Apply linear transformation
        t2 = t1 + 3                  # Add 3 to the output
        t3 = torch.clamp_min(t2, 0)  # Clamp to a minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp to a maximum of 6
        t5 = t4 / 6                  # Divide by 6
        return t5

# Initializing the model
model = Model()

# Generating input tensor
input_tensor = torch.randn(2, 10)  # Batch size of 2 and 10 features
output = model(input_tensor)

# Displaying the output
print(output)
