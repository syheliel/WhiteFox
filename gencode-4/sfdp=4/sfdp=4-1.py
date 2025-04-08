import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.query_layer = nn.Linear(embed_size, embed_size)
        self.key_layer = nn.Linear(embed_size, embed_size)
        self.value_layer = nn.Linear(embed_size, embed_size)
        self.num_heads = num_heads
        self.embed_size = embed_size

    def forward(self, query, key, value, attn_mask=None):
        # Compute the dot product of the query and key, and scale it
        scaled_dot_product = (self.query_layer(query) @ self.key_layer(key).transpose(-2, -1)) / (self.embed_size ** 0.5)
        
        if attn_mask is not None:
            scaled_dot_product += attn_mask
        
        # Apply softmax to the result
        attn_weights = F.softmax(scaled_dot_product, dim=-1)
        
        # Compute the dot product of the attention weights and the value
        output = attn_weights @ self.value_layer(value)
        return output

# Initialize the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Generate input tensors
batch_size = 1
sequence_length = 10  # Length of the input sequence
query = torch.randn(batch_size, sequence_length, embed_size)  # Query tensor
key = torch.randn(batch_size, sequence_length, embed_size)    # Key tensor
value = torch.randn(batch_size, sequence_length, embed_size)  # Value tensor
attn_mask = torch.zeros(batch_size, sequence_length, sequence_length)  # Attention mask, can be None or a tensor

# Forward pass
output = model(query, key, value, attn_mask)
print(output.shape)  # Output shape
