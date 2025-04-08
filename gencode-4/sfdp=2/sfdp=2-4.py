import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(embed_size, embed_size)
        self.key = torch.nn.Linear(embed_size, embed_size)
        self.value = torch.nn.Linear(embed_size, embed_size)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.inv_scale_factor = embed_size ** 0.5  # Set the inverse scale factor

    def forward(self, x):
        # Assuming x is of shape (batch_size, seq_length, embed_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by the inverse scale factor
        scaled_qk = qk.div(self.inv_scale_factor)
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Apply dropout to the softmax output
        dropout_qk = self.dropout(softmax_qk)
        
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(value)
        
        return output

# Initializing the model with arbitrary parameters
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads (not used here, but typical in multi-head attention)
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Generate input tensor
batch_size = 1
seq_length = 10  # Length of the input sequence
x = torch.randn(batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(x)

print(output.shape)  # Output shape
