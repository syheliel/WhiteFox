import torch

# Define the model
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)  # Compute attention weights
        attn_weight = torch.nn.functional.dropout(attn_weight, p=self.dropout_p)  # Apply dropout to the attention weights
        output = torch.bmm(attn_weight, value)  # Compute the output
        return output

# Initializing the model
embed_dim = 64  # Dimension of the embeddings
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_dim, dropout_p)

# Generate input tensors for the model
batch_size = 2  # Number of samples in the batch
seq_length = 10  # Sequence length

query = torch.randn(batch_size, seq_length, embed_dim)  # Shape: (batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)    # Shape: (batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)  # Shape: (batch_size, seq_length, embed_dim)

# Get the output from the model
output = model(query, key, value)

print("Output shape:", output.shape)  # Should be (batch_size, seq_length, embed_dim)
