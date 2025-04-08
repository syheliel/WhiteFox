import torch

class DropoutAndRandomModel(torch.nn.Module):
    def __init__(self, dropout_probability=0.5):
        super().__init__()
        self.dropout_probability = dropout_probability

    def forward(self, input_tensor):
        # Apply dropout to the input tensor
        t1 = torch.nn.functional.dropout(input_tensor, p=self.dropout_probability, training=self.training)
        # Generate a tensor with the same size as input_tensor filled with random numbers
        t2 = torch.rand_like(input_tensor)
        return t1, t2

# Initializing the model
model = DropoutAndRandomModel()

# Create input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output_t1, output_t2 = model(input_tensor)

print("Output of Dropout:", output_t1)
print("Output of Random Tensor:", output_t2)
