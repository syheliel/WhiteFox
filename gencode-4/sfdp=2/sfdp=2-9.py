import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, heads, dropout_p):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout_p = dropout_p
        
        self.values = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.keys = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.queries = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = torch.nn.Linear(embed_size, embed_size)
        
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))  # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(inv_scale_factor)  # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1)  # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)  # Apply dropout to the softmax output
        output = dropout_qk.matmul(value)  # Compute the dot product of the dropout output and the value tensor
        return self.fc_out(output)

# Initializing the model parameters
embed_size = 64  # Size of the embedding
heads = 8  # Number of attention heads
dropout_p = 0.1  # Dropout probability
inv_scale_factor = embed_size ** 0.5  # Inverse scale factor

# Initializing the model
model = AttentionModel(embed_size, heads, dropout_p)

# Generating input tensors
batch_size = 1
seq_length = 10  # Length of the sequence
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Forward pass
output = model(query, key, value, inv_scale_factor)

# Output shape
print(output.shape)  # Should be (batch_size, seq_length, embed_size)
