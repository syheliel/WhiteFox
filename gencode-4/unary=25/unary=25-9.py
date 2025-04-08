import torch

# Model Definition
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
        self.negative_slope = negative_slope
    
    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 > 0  # Create boolean tensor
        t3 = t1 * self.negative_slope  # Apply negative slope
        t4 = torch.where(t2, t1, t3)  # Implement Leaky ReLU
        return t4

# Initializing the model
model = LeakyReLUModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 features
output_tensor = model(input_tensor)

# Display the output
print("Output Tensor:", output_tensor)
