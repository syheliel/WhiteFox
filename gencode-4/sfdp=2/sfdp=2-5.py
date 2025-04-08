import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, dropout_p):
        super().__init__()
        self.query_layer = torch.nn.Linear(embed_size, embed_size)
        self.key_layer = torch.nn.Linear(embed_size, embed_size)
        self.value_layer = torch.nn.Linear(embed_size, embed_size)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.inv_scale_factor = embed_size ** 0.5  # Inverse scale factor (sqrt of embed_size)

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))  # Compute the dot product of query and key tensors
        scaled_qk = qk / self.inv_scale_factor  # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1)  # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk)  # Apply dropout to the softmax output
        output = dropout_qk.matmul(value)  # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
embed_size = 64  # Size of the embeddings
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, dropout_p)

# Generating input tensors
batch_size = 2  # Number of samples in a batch
seq_length = 10  # Length of the sequences
query = torch.randn(batch_size, seq_length, embed_size)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_size)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_size)  # Value tensor

# Forward pass through the model
output = model(query, key, value)

print("Output shape:", output.shape)  # Should be (batch_size, seq_length, embed_size)
