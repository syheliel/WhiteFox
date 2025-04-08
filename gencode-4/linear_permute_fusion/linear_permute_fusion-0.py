import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with input features = 10 and output features = 20
        self.linear = torch.nn.Linear(10, 20)

    def forward(self, x1):
        t1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)  # Apply linear transformation
        t2 = t1.permute(0, 2, 1)  # Assuming x1 has 3 dimensions: (batch_size, seq_length, features)
        return t2

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(5, 3, 10)  # Example input tensor with shape (batch_size=5, seq_length=3, features=10)
output = model(x1)

print("Output shape:", output.shape)
