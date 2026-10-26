from pathlib import Path

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


sample = "the cat"

encoded = encode(sample)
decoded = decode(encoded)

print("Vocabulary:", vocabulary)
print("Vocabulary size:", vocab_size)

print("\nOriginal text:", sample)
print("Encoded IDs:", encoded)
print("Decoded text:", decoded)

assert decoded == sample