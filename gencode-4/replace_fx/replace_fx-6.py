import torch

class RandomModel(torch.nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(self, input_tensor):
        # Apply dropout to the input tensor
        t1 = torch.nn.functional.dropout(input_tensor, p=self.dropout_prob, training=self.training)
        # Generate a tensor with the same size as input_tensor filled with random numbers
        t2 = torch.rand_like(input_tensor)
        return t1, t2

# Initializing the model
model = RandomModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass
output_t1, output_t2 = model(input_tensor)

print("Output from Dropout:", output_t1)
print("Random Tensor:", output_t2)
