import torch

# Model definition
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Compute attention weights
        attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
        # Apply dropout to the attention weights
        attn_weight = self.dropout(attn_weight)
        # Compute the output
        output = torch.bmm(attn_weight, value)
        return output

# Parameters
embed_size = 64  # Embedding size
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability

# Initializing the model
model = AttentionModel(embed_size, num_heads, dropout_p)

# Example inputs to the model
batch_size = 2
seq_length = 10

# Creating random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Passing the inputs through the model
output = model(query, key, value)

# Output shape
print(output.shape)  # Expected shape: (batch_size, seq_length, embed_size)
