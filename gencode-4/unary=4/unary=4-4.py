import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(128, 64)  # Linear transformation from 128 to 64 dimensions

    def forward(self, input_tensor):
        l1 = self.linear(input_tensor)  # Apply pointwise linear transformation
        l2 = l1 * 0.5                   # Multiply the output by 0.5
        l3 = l1 * 0.7071067811865476    # Multiply the output by 0.7071067811865476
        l4 = torch.erf(l3)              # Apply the error function
        l5 = l4 + 1                     # Add 1 to the output of the error function
        l6 = l2 * l5                    # Multiply the output by the output of the error function
        return l6

# Initializing the model
model = Model()

# Generating input tensor for the model
input_tensor = torch.randn(1, 128)  # A batch with 1 sample and 128 features
output = model(input_tensor)

# Print the output shape
print(output.shape)  # Should be [1, 64]
