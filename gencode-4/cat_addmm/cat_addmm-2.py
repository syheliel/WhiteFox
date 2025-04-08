import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define two weight matrices for matrix multiplication
        self.mat1 = torch.nn.Parameter(torch.randn(3, 5))  # Input features to hidden features
        self.mat2 = torch.nn.Parameter(torch.randn(5, 4))  # Hidden features to output features

    def forward(self, x):
        # Perform matrix multiplication and add to input
        t1 = torch.addmm(x, self.mat1.t(), self.mat2)
        # Concatenate along the last dimension
        t2 = torch.cat([t1], dim=-1)
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)  # Single sample with 3 features
output = m(x1)

print("Output shape:", output.shape)
