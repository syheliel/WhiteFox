import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features

    def forward(self, x):
        t1 = self.linear(x)        # Apply linear transformation
        t2 = t1 + 3                # Add 3
        t3 = torch.clamp_min(t2, 0)  # Clamp to minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp to maximum of 6
        t5 = t4 / 6                # Divide by 6
        return t5

# Initializing the model
model = Model()

# Input to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 input features
output = model(input_tensor)

# Output shape and value
print("Output shape:", output.shape)
print("Output tensor:", output)
