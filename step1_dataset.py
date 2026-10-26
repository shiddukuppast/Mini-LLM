from pathlib import Path

file_path = Path("data/input.txt")

text = file_path.read_text(encoding="utf-8")

print("Dataset:")
print(text)

print("Number of characters:", len(text))
print("Number of lines:", len(text.splitlines()))
print("Unique characters:", len(set(text)))