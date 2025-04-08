import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(embed_size, embed_size)
        self.key = torch.nn.Linear(embed_size, embed_size)
        self.value = torch.nn.Linear(embed_size, embed_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.inv_scale_factor = embed_size ** 0.5  # Inverse scale factor for dot-product scaling

    def forward(self, query, key, value):
        qk = torch.matmul(self.query(query), self.key(key).transpose(-2, -1))  # Dot product of query and key
        scaled_qk = qk / self.inv_scale_factor  # Scale dot product
        softmax_qk = scaled_qk.softmax(dim=-1)  # Apply softmax
        dropout_qk = self.dropout(softmax_qk)  # Apply dropout
        output = torch.matmul(dropout_qk, self.value(value))  # Dot product with value tensor
        return output

# Initialize the model
embed_size = 64  # Embedding size
num_heads = 8    # Number of attention heads (not used in this simple model)
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10  # Number of tokens in the sequence
query_tensor = torch.randn(batch_size, seq_length, embed_size)  # Query tensor
key_tensor = torch.randn(batch_size, seq_length, embed_size)    # Key tensor
value_tensor = torch.randn(batch_size, seq_length, embed_size)  # Value tensor

# Forward pass
output_tensor = model(query_tensor, key_tensor, value_tensor)

# Print the output shape
print(output_tensor.shape)  # Expected shape: (batch_size, seq_length, embed_size)
