import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, inv_scale_factor):
        q = self.query(query).view(-1, query.size(1), self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key(key).view(-1, key.size(1), self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)  # Permute the dimensions of the key tensor
        v = self.value(value).view(-1, value.size(1), self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        
        t1 = torch.matmul(q, k.transpose(-2, -1))  # Perform matrix multiplication between the query tensor and the transposed key tensor
        t2 = t1.div(inv_scale_factor)  # Divide the result by an inverse scale factor
        t3 = t2.softmax(dim=-1)  # Apply softmax function to the result
        t4 = nn.functional.dropout(t3, p=self.dropout_p)  # Apply dropout to the result
        output = t4.matmul(v)  # Perform matrix multiplication between the result and the value tensor
        
        return output

# Initialize the model parameters
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability

# Initialize the model
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Example input tensors
batch_size = 2   # Number of samples in a batch
seq_length = 10  # Length of the sequence

query_input = torch.randn(batch_size, seq_length, embed_dim)  # Query tensor
key_input = torch.randn(batch_size, seq_length, embed_dim)    # Key tensor
value_input = torch.randn(batch_size, seq_length, embed_dim)  # Value tensor
inv_scale_factor = embed_dim ** 0.5  # Inverse scale factor for scaling

# Forward pass through the model
output = model(query_input, key_input, value_input, inv_scale_factor)

# Print the output shape
print(output.shape)  # Expected: (batch_size, num_heads, seq_length, embed_dim // num_heads)
