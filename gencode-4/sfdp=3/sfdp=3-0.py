import torch

# Model definition
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, dropout_p=0.1):
        super().__init__()
        self.query = torch.nn.Linear(embed_size, embed_size)
        self.key = torch.nn.Linear(embed_size, embed_size)
        self.value = torch.nn.Linear(embed_size, embed_size)
        self.scale_factor = embed_size ** -0.5  # Scaling factor for the dot product
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, queries, keys, values):
        qk = torch.matmul(self.query(queries), self.key(keys).transpose(-2, -1))  # Compute the dot product of query and transposed key
        scaled_qk = qk.mul(self.scale_factor)  # Scale the dot product by a factor
        softmax_qk = scaled_qk.softmax(dim=-1)  # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk)  # Apply dropout to the softmax output
        output = dropout_qk.matmul(self.value(values))  # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
embed_size = 64  # Size of the embedding
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10  # Sequence length
queries = torch.randn(batch_size, seq_length, embed_size)
keys = torch.randn(batch_size, seq_length, embed_size)
values = torch.randn(batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(queries, keys, values)

print("Output shape:", output.shape)
