import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, inv_scale_factor):
        q = self.query(x).permute(0, 2, 1)  # Permute the dimensions of the query tensor
        k = self.key(x).permute(0, 2, 1)    # Permute the dimensions of the key tensor
        v = self.value(x).permute(0, 2, 1)  # Permute the dimensions of the value tensor
        
        t1 = torch.matmul(q, k.transpose(-2, -1))  # Matrix multiplication with transposed key tensor
        t2 = t1.div(inv_scale_factor)                # Divide by inverse scale factor
        t3 = t2.softmax(dim=-1)                      # Apply softmax
        t4 = self.dropout(t3)                        # Apply dropout
        output = t4.matmul(v)                        # Matrix multiplication with value tensor
        
        return output

# Model parameters
embed_dim = 64     # Embedding dimension
num_heads = 8      # Number of attention heads (not used in this example)
dropout_p = 0.1    # Dropout probability

# Initializing the model
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Create a dummy input tensor
# Example input shape (batch_size, sequence_length, embedding_dim)
batch_size = 1
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, embed_dim)

# Inverse scale factor
inv_scale_factor = 1.0 / (embed_dim ** 0.5)

# Forward pass through the model
output = model(input_tensor, inv_scale_factor)

# Print the output shape
print(output.shape)  # Expected shape: (batch_size, seq_length, embed_dim)
