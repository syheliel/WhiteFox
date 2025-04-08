import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.inv_scale = embed_size ** 0.5
        
        # Define linear layers for query, key, and value
        self.query_linear = torch.nn.Linear(embed_size, embed_size)
        self.key_linear = torch.nn.Linear(embed_size, embed_size)
        self.value_linear = torch.nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Permute the dimensions
        q = self.query_linear(query).permute(0, 2, 1, 3)  # (batch_size, seq_length, heads, head_dim)
        k = self.key_linear(key).permute(0, 2, 1, 3)      # (batch_size, seq_length, heads, head_dim)
        v = self.value_linear(value).permute(0, 2, 1, 3)  # (batch_size, seq_length, heads, head_dim)

        # Compute scaled dot-product attention
        attention = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, heads, seq_length, seq_length)
        scaled_attention = attention / self.inv_scale       # Scale the attention
        attention_weights = scaled_attention.softmax(dim=-1) # Apply softmax to get weights

        # Compute the weighted sum of the value tensor
        output = attention_weights.matmul(v)                # (batch_size, heads, seq_length, head_dim)
        return output

# Initializing the model
embed_size = 64  # Embedding size
heads = 8       # Number of attention heads
model = AttentionModel(embed_size, heads)

# Inputs to the model
batch_size = 1
seq_length = 10

# Create random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Forward pass
output = model(query, key, value)
