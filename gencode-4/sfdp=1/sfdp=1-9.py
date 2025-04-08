import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(embed_size, embed_size)
        self.key = torch.nn.Linear(embed_size, embed_size)
        self.value = torch.nn.Linear(embed_size, embed_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.scale = embed_size ** 0.5  # Scaling factor for dot product

    def forward(self, x):
        # Assume x is of shape (batch_size, seq_length, embed_size)
        q = self.query(x)  # Shape: (batch_size, seq_length, embed_size)
        k = self.key(x)    # Shape: (batch_size, seq_length, embed_size)
        v = self.value(x)  # Shape: (batch_size, seq_length, embed_size)

        qk = torch.matmul(q, k.transpose(-2, -1))  # Shape: (batch_size, seq_length, seq_length)
        scaled_qk = qk / self.scale  # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)  # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk)  # Apply dropout to the softmax output
        output = torch.matmul(dropout_qk, v)  # Compute the dot product of the dropout output and value tensor

        return output

# Initializing the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads (not used in this simple model)
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10
x = torch.randn(batch_size, seq_length, embed_size)  # Random input tensor

# Forward pass
output = model(x)
print(output.shape)  # Output shape should be (batch_size, seq_length, embed_size)
