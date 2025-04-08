import torch

# Define the model
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        # Define layers
        self.query_layer = torch.nn.Linear(embed_size, embed_size)
        self.key_layer = torch.nn.Linear(embed_size, embed_size)
        self.value_layer = torch.nn.Linear(embed_size, embed_size)
        
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Calculate attention weights
        attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
        
        # Apply dropout to the attention weights
        attn_weight = self.dropout(attn_weight)
        
        # Compute the output
        output = torch.bmm(attn_weight, value)
        return output

# Initialize the model parameters
embed_size = 64  # Embedding size
num_heads = 8    # Number of heads (not used in this basic implementation)
dropout_p = 0.1  # Dropout probability

# Create an instance of the model
model = AttentionModel(embed_size, num_heads, dropout_p)

# Generate input tensors
batch_size = 10   # Number of samples in the batch
seq_length = 20   # Length of the sequence

# Create random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(query, key, value)

# Output the shape of the result
print("Output shape:", output.shape)
