from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. CONFIGURATION
# ==========================================

torch.manual_seed(42)

device = torch.device("cpu")

print("Using device:", device)


# ==========================================
# 2. LOAD DATASET
# ==========================================

text = Path("data/input.txt").read_text(encoding="utf-8")

print("Total characters:", len(text))


# ==========================================
# 3. BUILD VOCABULARY
# ==========================================

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


# Convert the complete dataset into token IDs.
data = torch.tensor(encode(text), dtype=torch.long)

print("Vocabulary size:", vocab_size)
print("Encoded dataset shape:", data.shape)


# ==========================================
# 4. CREATE INPUT AND TARGET PAIRS
# ==========================================

# Each character predicts the character that follows it.

x = data[:-1]
y = data[1:]

print("Input shape:", x.shape)
print("Target shape:", y.shape)

print("\nExample training pairs:")

for i in range(min(10, len(x))):
    input_character = decode([x[i].item()])
    target_character = decode([y[i].item()])

    print(
        f"Input: {repr(input_character):8}"
        f"Target: {repr(target_character)}"
    )


# ==========================================
# 5. DEFINE THE BIGRAM MODEL
# ==========================================

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Every previous token has a row of trainable scores
        # for all possible next tokens.
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):

        # idx shape: (batch_size, sequence_length)
        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:

            batch_size, sequence_length, channels = logits.shape

            logits = logits.reshape(
                batch_size * sequence_length,
                channels
            )

            targets = targets.reshape(
                batch_size * sequence_length
            )

            loss = F.cross_entropy(logits, targets)

        return logits, loss


# ==========================================
# 6. INITIALIZE MODEL
# ==========================================

model = BigramLanguageModel(vocab_size).to(device)

print("\nModel:")
print(model)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Trainable parameters:", parameter_count)


# ==========================================
# 7. CONFIGURE OPTIMIZER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01
)


# ==========================================
# 8. TRAINING LOOP
# ==========================================

epochs = 1000

model.train()

for epoch in range(epochs):

    # Forward pass
    logits, loss = model(
        x.unsqueeze(0),
        y.unsqueeze(0)
    )

    # Clear old gradients
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )


# ==========================================
# 9. GENERATE TEXT
# ==========================================

@torch.no_grad()
def generate(start_text, max_new_tokens=100, temperature=1.0):

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    if not start_text:
        raise ValueError("Start text cannot be empty.")

    if any(character not in stoi for character in start_text):
        raise ValueError(
            "Start text contains characters absent from the dataset."
        )

    model.eval()

    idx = torch.tensor(
        [encode(start_text)],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):

        # Obtain the logits for the last character.
        logits, _ = model(idx)

        logits = logits[:, -1, :]

        # Control the sharpness of the probability distribution.
        logits = logits / temperature

        # Convert scores into probabilities.
        probabilities = F.softmax(logits, dim=-1)

        # Sample the next character.
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Append the sampled character.
        idx = torch.cat(
            [idx, next_token],
            dim=1
        )

    return decode(idx[0].tolist())


# ==========================================
# 10. TEST THE TRAINED MODEL
# ==========================================

print("\n" + "=" * 40)
print("GENERATED TEXT")
print("=" * 40)

print(generate("the cat", max_new_tokens=100))

print("\nTraining completed!")
