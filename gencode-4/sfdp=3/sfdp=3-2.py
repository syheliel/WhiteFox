import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(embed_size, embed_size)
        self.key = torch.nn.Linear(embed_size, embed_size)
        self.value = torch.nn.Linear(embed_size, embed_size)
        self.scale_factor = 1.0 / (embed_size ** 0.5)  # Scale factor for dot product
        self.dropout = torch.nn.Dropout(dropout_p)  # Dropout layer

    def forward(self, x):
        q = self.query(x)  # Query
        k = self.key(x)    # Key
        v = self.value(x)  # Value

        qk = torch.matmul(q, k.transpose(-2, -1))  # Compute the dot product of the query and transposed key
        scaled_qk = qk.mul(self.scale_factor)        # Scale the dot product by a factor
        softmax_qk = scaled_qk.softmax(dim=-1)      # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk)       # Apply dropout to the softmax output
        output = dropout_qk.matmul(v)                # Compute the dot product of the dropout output and value

        return output

# Initializing the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads (not used in this simplified model)
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Inputs to the model
batch_size = 1  # Number of samples in the batch
sequence_length = 10  # Number of tokens in the sequence
x = torch.randn(batch_size, sequence_length, embed_size)  # Random input tensor
output = model(x)

print("Output shape:", output.shape)  # Print the shape of the output

x = torch.randn(batch_size, sequence_length, embed_size)  # Random input tensor
