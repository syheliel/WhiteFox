import torch
import math

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.query_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.key_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.value_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        q = self.query_layer(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key_layer(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.value_layer(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        q = q / math.sqrt(q.size(-1))                     # Scale the query tensor
        div = q @ k.transpose(-2, -1)                     # Matrix multiplication
        div = div.to(torch.float32)                        # Convert the result to float32
        attn_weight = torch.softmax(div, dim=-1)          # Apply softmax
        attn_weight = self.dropout(attn_weight)            # Apply dropout
        attn_weight = attn_weight.to(torch.float16)       # Convert to float16
        output = attn_weight @ v                           # Matrix multiplication with value tensor
        return output

# Initializing the model with specified embedding dimension and dropout probability
embed_dim = 64  # Example embedding dimension
dropout_p = 0.1 # Example dropout probability
model = AttentionModel(embed_dim, dropout_p)

# Inputs to the model (batch_size, seq_length, embed_dim)
batch_size = 1
seq_length = 10  # Example sequence length
query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)

# Running the model with the input tensors
output = model(query, key, value)
print(output.shape)  # Should give the shape of the output tensor
