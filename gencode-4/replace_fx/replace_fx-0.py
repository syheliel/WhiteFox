import torch

# Model
class CustomModel(torch.nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.linear1 = torch.nn.Linear(10, 20)  # Example linear layer
        self.linear2 = torch.nn.Linear(20, 10)  # Another linear layer

    def forward(self, input_tensor):
        t1 = torch.nn.functional.dropout(input_tensor, p=self.dropout_prob, training=self.training)  # Apply dropout
        t2 = torch.rand_like(input_tensor)  # Generate a tensor with random values
        t3 = self.linear1(t1)  # Pass through linear layer
        t4 = self.linear2(t3 + t2)  # Add the random tensor to the output of the first layer and pass through second layer
        return t4

# Initializing the model
model = CustomModel()

# Inputs to the model
input_tensor = torch.randn(5, 10)  # Batch size of 5, input feature size of 10
output = model(input_tensor)

print("Output shape:", output.shape)
