import torch

class MatrixAddConcatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)  # First linear layer
        self.linear2 = torch.nn.Linear(10, 5)  # Second linear layer

    def forward(self, x):
        t1 = torch.addmm(x, self.linear1.weight, self.linear1.weight.t())  # Matrix multiplication and addition
        t2 = torch.cat([t1], dim=1)  # Concatenate along dimension 1
        return t2

# Initializing the model
model = MatrixAddConcatModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10
output = model(input_tensor)

print("Output Shape:", output.shape)  # To verify output shape
