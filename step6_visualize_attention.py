from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Import the model architecture.
import step5_transformer as training_module

# ==========================================
# 1. CONFIGURATION
# ==========================================

CHECKPOINT_PATH = Path("mini_gpt_checkpoint.pth")

device = torch.device("cpu")


# ==========================================
# 2. CHECK CHECKPOINT
# ==========================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
    )


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

# This is a checkpoint created by our own training script.
# weights_only=True is appropriate for the state dictionary
# and the basic configuration values stored in this checkpoint.

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

# Restore the configuration used during training.
training_module.BLOCK_SIZE = checkpoint["block_size"]
training_module.EMBEDDING_DIM = checkpoint["embedding_dim"]
training_module.NUM_HEADS = checkpoint["num_heads"]
training_module.NUM_LAYERS = checkpoint["num_layers"]
training_module.DROPOUT = checkpoint["dropout"]
training_module.vocab_size = checkpoint["vocab_size"]

# Construct the model using the saved configuration.
model = training_module.MiniGPT().to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Trained MiniGPT loaded successfully!")


# ==========================================
# 4. PREPARE INPUT TEXT
# ==========================================

vocabulary = checkpoint["vocabulary"]

stoi = {
    character: index
    for index, character in enumerate(vocabulary)
}

itos = {
    index: character
    for character, index in stoi.items()
}


def encode(text):
    return [stoi[character] for character in text]


sample_text = "cat is sleeping"

if any(character not in stoi for character in sample_text):
    raise ValueError(
        "The sample text contains characters outside the vocabulary."
    )

tokens = list(sample_text)

input_ids = torch.tensor(
    [encode(sample_text)],
    dtype=torch.long,
    device=device
)

print("Input text:", sample_text)
print("Number of characters:", len(tokens))


# ==========================================
# 5. EXTRACT ATTENTION WEIGHTS
# ==========================================

with torch.no_grad():
    logits, _ = model(input_ids)

# Get the first Transformer block.
first_block = model.blocks[0]

# Get its first attention head.
first_head = first_block.attention.heads[0]

# Shape: (batch_size, sequence_length, sequence_length)
attention_weights = (
    first_head.last_attention_weights[0]
    .cpu()
    .numpy()
)

print("Attention matrix shape:", attention_weights.shape)


# ==========================================
# 6. VISUALIZE ATTENTION
# ==========================================

plt.figure(figsize=(12, 9))

plt.imshow(
    attention_weights,
    interpolation="nearest",
    aspect="auto"
)

plt.xticks(
    range(len(tokens)),
    tokens,
    fontsize=10
)

plt.yticks(
    range(len(tokens)),
    tokens,
    fontsize=10
)

plt.xlabel("Key position: token being attended to")
plt.ylabel("Query position: token using context")

plt.title(
    "MiniGPT Attention Weights — First Head\n"
    f'Input: "{sample_text}"'
)

plt.colorbar(label="Attention weight")

plt.tight_layout()

# Save a copy of the visualization.
plt.savefig(
    "attention_heatmap.png",
    dpi=150,
    bbox_inches="tight"
)

print("Heatmap saved to attention_heatmap.png")

plt.show()
