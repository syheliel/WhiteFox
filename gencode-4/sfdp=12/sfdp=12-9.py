import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        # Compute attention weights
        attn_weights = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
        # Apply dropout to the attention weights
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout_p)
        # Compute the output
        output = torch.bmm(attn_weights, value)
        return output

# Initializing the model
embed_dim = 64  # Embedding dimension for queries, keys, and values
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_dim, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10  # Number of tokens in the sequence
query = torch.randn(batch_size, seq_length, embed_dim)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_dim)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_dim)  # Value tensor

# Running the model
output = model(query, key, value)

# Output shape
print("Output shape:", output.shape)
