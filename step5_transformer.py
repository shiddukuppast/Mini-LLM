from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. CONFIGURATION
# ==========================================

torch.manual_seed(42)

device = torch.device("cpu")

BATCH_SIZE = 16
BLOCK_SIZE = 16
MAX_ITERS = 2000
EVAL_INTERVAL = 200
LEARNING_RATE = 0.001

EMBEDDING_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 1
DROPOUT = 0.1


# ==========================================
# 2. LOAD AND TOKENIZE DATA
# ==========================================

text = Path("data/input.txt").read_text(encoding="utf-8")

vocabulary = sorted(set(text))
vocab_size = len(vocabulary)

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


def decode(token_ids):
    return "".join(itos[index] for index in token_ids)


data = torch.tensor(
    encode(text),
    dtype=torch.long
)

# Split the data into training and validation portions.
split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]

print("Device:", device)
print("Vocabulary size:", vocab_size)
print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))


# ==========================================
# 3. CREATE BATCHES
# ==========================================

def get_batch(split):

    source = train_data if split == "train" else val_data

    if len(source) <= BLOCK_SIZE:
        raise ValueError(
            f"{split} dataset must contain more than "
            f"{BLOCK_SIZE} characters."
        )

    # Random starting positions for sequences.
    starts = torch.randint(
        0,
        len(source) - BLOCK_SIZE,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        source[start:start + BLOCK_SIZE]
        for start in starts
    ])

    y = torch.stack([
        source[start + 1:start + BLOCK_SIZE + 1]
        for start in starts
    ])

    return x.to(device), y.to(device)


# ==========================================
# 4. SELF-ATTENTION HEAD
# ==========================================

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        batch_size, sequence_length, channels = x.shape

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        scores = query @ key.transpose(-2, -1)

        scores = scores * (key.shape[-1] ** -0.5)

        scores = scores.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)

        # Save the original attention distribution.
        self.last_attention_weights = weights.detach()

        # Apply dropout only to the weights used for computation.
        weights = self.dropout(weights)

        output = weights @ value

        return output


# ==========================================
# 5. MULTI-HEAD SELF-ATTENTION
# ==========================================

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            num_heads * head_size,
            EMBEDDING_DIM
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        output = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        output = self.projection(output)

        return self.dropout(output)


# ==========================================
# 6. FEED-FORWARD NETWORK
# ==========================================

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 4 * EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 7. TRANSFORMER BLOCK
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        head_size = EMBEDDING_DIM // NUM_HEADS

        self.attention = MultiHeadAttention(
            NUM_HEADS,
            head_size
        )

        self.feed_forward = FeedForward()

        self.norm1 = nn.LayerNorm(EMBEDDING_DIM)
        self.norm2 = nn.LayerNorm(EMBEDDING_DIM)

    def forward(self, x):

        # Residual connection around attention.
        x = x + self.attention(self.norm1(x))

        # Residual connection around feed-forward network.
        x = x + self.feed_forward(self.norm2(x))

        return x


# ==========================================
# 8. MINI GPT MODEL
# ==========================================

class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM
        )

        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            EMBEDDING_DIM
        )

        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(NUM_LAYERS)]
        )

        self.final_norm = nn.LayerNorm(EMBEDDING_DIM)

        self.language_head = nn.Linear(
            EMBEDDING_DIM,
            vocab_size
        )

    def forward(self, idx, targets=None):

        batch_size, sequence_length = idx.shape

        token_embeddings = self.token_embedding(idx)

        positions = torch.arange(
            sequence_length,
            device=idx.device
        )

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        x = self.blocks(x)

        x = self.final_norm(x)

        logits = self.language_head(x)

        loss = None

        if targets is not None:

            logits = logits.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0):

        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the most recent context.
            idx_context = idx[:, -BLOCK_SIZE:]

            logits, _ = self(idx_context)

            # Obtain the next-token logits.
            logits = logits.reshape(
                idx.shape[0],
                idx_context.shape[1],
                vocab_size
            )

            logits = logits[:, -1, :]

            logits = logits / temperature

            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

        return idx


# ==========================================
# 9. EVALUATE LOSS
# ==========================================

model = MiniGPT().to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel parameters:", parameter_count)


@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = torch.zeros(10)

        for k in range(10):

            x, y = get_batch(split)

            _, loss = model(x, y)

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results

def main():

    # ==========================================
    # 10. TRAIN THE MODEL
    # ==========================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nStarting training...")

    for iteration in range(MAX_ITERS):

        if iteration % EVAL_INTERVAL == 0:

            losses = estimate_loss()

            print(
                f"Step {iteration:4d} | "
                f"Train loss: {losses['train']:.4f} | "
                f"Validation loss: {losses['val']:.4f}"
            )

        x, y = get_batch("train")

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()


    # ==========================================
    # 11. GENERATE TEXT
    # ==========================================

    model.eval()

    start_text = "the cat"

    start_ids = torch.tensor(
        [encode(start_text)],
        dtype=torch.long,
        device=device
    )

    generated_ids = model.generate(
        start_ids,
        max_new_tokens=200,
        temperature=0.8
    )

    generated_text = decode(generated_ids[0].tolist())

    print("\n" + "=" * 50)
    print("GENERATED TEXT")
    print("=" * 50)

    print(generated_text)

    print("\nTraining completed!")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "vocabulary": vocabulary,
            "block_size": BLOCK_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        "mini_gpt_checkpoint.pth"
    )

    print("Model checkpoint saved!")
    
if __name__ == "__main__":
    main()