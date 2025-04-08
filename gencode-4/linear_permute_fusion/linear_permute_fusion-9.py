import torch

class LinearPermuteModel(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(LinearPermuteModel, self).__init__()
        self.linear = torch.nn.Linear(input_features, output_features)

    def forward(self, x):
        t1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)  # Apply linear transformation
        t2 = t1.permute(0, 2, 1)  # Permute the output tensor (assuming input was of shape [batch_size, seq_length, features])
        return t2

# Initializing the model
input_features = 4  # Number of input features
output_features = 6  # Number of output features
model = LinearPermuteModel(input_features, output_features)

# Generate input tensor
# Assuming the input tensor has shape (batch_size, seq_length, input_features)
batch_size = 2
seq_length = 3
input_tensor = torch.randn(batch_size, seq_length, input_features)

# Forward pass
output = model(input_tensor)
