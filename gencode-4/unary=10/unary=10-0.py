import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 + 3          # Add 3 to the output
        t3 = torch.clamp_min(t2, 0)  # Clamp to a minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp to a maximum of 6
        t5 = t4 / 6          # Normalize by dividing by 6
        return t5

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Create a random input tensor with shape (1, 10)
output = model(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("\nOutput Tensor:")
print(output)
