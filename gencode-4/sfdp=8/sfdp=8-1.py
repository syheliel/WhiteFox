import torch
import torch.nn as nn
import math

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Linear layers to project inputs to query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, head_dim)
        k = self.key_layer(key).permute(0, 2, 1, 3)      # (batch_size, seq_length, num_heads, head_dim)
        v = self.value_layer(value).permute(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, head_dim)

        # Scale the query tensor
        q = q / math.sqrt(q.size(-1))

        # Matrix multiplication between the query and the transposed key
        div = q @ k.transpose(-2, -1)  # (batch_size, num_heads, seq_length, seq_length)

        # Convert result to float32
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = torch.softmax(div, dim=-1)

        # Apply dropout to the softmax result
        attn_weight = self.dropout(attn_weight)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Matrix multiplication between the attention weights and the value tensor
        attn_output = attn_weight @ v  # (batch_size, num_heads, seq_length, head_dim)
        
        return attn_output

# Initialize the model
embed_dim = 64  # Size of the embedding
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Generate input tensors
batch_size = 2
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, embed_dim)  # Input shape: (batch_size, seq_length, embed_dim)

# Forward pass through the model
output = model(input_tensor, input_tensor, input_tensor)

print("Output shape:", output.shape)  # Output shape should be (batch_size, num_heads, seq_length, head_dim)
