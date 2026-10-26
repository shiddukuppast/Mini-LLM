import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ------------------------------------------
# 1. CREATE A SMALL INPUT SEQUENCE
# ------------------------------------------

# Batch size = 1
# Sequence length = 4
# Embedding dimension = 8

x = torch.randn(1, 4, 8)

print("Input shape:", x.shape)


# ------------------------------------------
# 2. CREATE QUERY, KEY AND VALUE LAYERS
# ------------------------------------------

embedding_dim = 8

query_layer = nn.Linear(embedding_dim, embedding_dim)
key_layer = nn.Linear(embedding_dim, embedding_dim)
value_layer = nn.Linear(embedding_dim, embedding_dim)


# ------------------------------------------
# 3. COMPUTE QUERIES, KEYS AND VALUES
# ------------------------------------------

Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

print("Query shape:", Q.shape)
print("Key shape:", K.shape)
print("Value shape:", V.shape)


# ------------------------------------------
# 4. CALCULATE ATTENTION SCORES
# ------------------------------------------

scores = Q @ K.transpose(-2, -1)

print("Attention scores shape:", scores.shape)


# ------------------------------------------
# 5. SCALE THE SCORES
# ------------------------------------------

scores = scores / (embedding_dim ** 0.5)


# ------------------------------------------
# 6. APPLY CAUSAL MASKING
# ------------------------------------------

sequence_length = x.shape[1]

# True means the position is allowed to be attended to.
causal_mask = torch.tril(
    torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool
    )
)

# Prevent each position from attending to future positions.
scores = scores.masked_fill(
    ~causal_mask,
    float("-inf")
)


# ------------------------------------------
# 7. CONVERT SCORES INTO ATTENTION WEIGHTS
# ------------------------------------------

attention_weights = F.softmax(scores, dim=-1)

print("\nAttention weights:")
print(attention_weights[0].detach())


# ------------------------------------------
# 8. COMPUTE THE ATTENTION OUTPUT
# ------------------------------------------

output = attention_weights @ V

print("\nOutput shape:", output.shape)
print("Output:")
print(output.detach())


# ------------------------------------------
# 9. VERIFY CAUSAL MASKING
# ------------------------------------------

print("\nFuture-token attention weights:")

for i in range(sequence_length):
    print(f"Position {i}:", attention_weights[0, i].detach())


# ------------------------------------------
# 10. VERIFY THAT GRADIENTS WORK
# ------------------------------------------

loss = output.square().mean()

loss.backward()

print("\nQuery layer gradient exists:",
      query_layer.weight.grad is not None)

print("Key layer gradient exists:",
      key_layer.weight.grad is not None)

print("Value layer gradient exists:",
      value_layer.weight.grad is not None)