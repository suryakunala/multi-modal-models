# Tokenizer training script

# Imports
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace

DIR_PATH_TO_DATA = Path("")
VOCAB_SIZE = 10_000
MIN_FREQUENCY = 10
PATH_TO_SAVE_TOKENIZER = Path("")


def train_tokenizer(tokenizer, file_paths, vocab_size, min_frequency, dest_path):
    tokenizer.train(
        files=file_paths,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
            "<mask>"
        ]
    )
    tokenizer.save_model(dest_path)

tokenizer = ByteLevelBPETokenizer()
tokenizer.pre_tokenizer = Whitespace()
print("Starting Tokenizer training")
train_tokenizer(
    tokenizer=tokenizer,
    file_paths=[str(path) for path in DIR_PATH_TO_DATA.rglob("*.*")],
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    dest_path=PATH_TO_SAVE_TOKENIZER
)
print(f"Tokenizer trained and saved at {PATH_TO_SAVE_TOKENIZER}")

