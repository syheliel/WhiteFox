import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.query_linear = torch.nn.Linear(embed_size, embed_size)
        self.key_linear = torch.nn.Linear(embed_size, embed_size)
        self.value_linear = torch.nn.Linear(embed_size, embed_size)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Compute attention weights
        attn_weights = torch.bmm(self.query_linear(query), self.key_linear(key).transpose(1, 2)).softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)  # Apply dropout to the attention weights
        output = torch.bmm(attn_weights, self.value_linear(value))  # Compute the output
        return output

# Initialize the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads (not used here but can be for multi-head attention)
dropout_p = 0.1  # Dropout probability

model = AttentionModel(embed_size, num_heads, dropout_p)

# Generate input tensors
batch_size = 10  # Number of samples in a batch
seq_length = 20  # Length of the input sequences

# Input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(query, key, value)
