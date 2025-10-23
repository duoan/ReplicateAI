import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from tokenizer import BaseTokenizer


class ToyDataset(torch.utils.data.Dataset):

    def __init__(self, vocab_size=20, seq_len=10, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.samples = torch.randint(1, vocab_size, (num_samples, seq_len))

    def __getitem__(self, index):
        x = self.samples[index]
        y = torch.cat([x[1:], torch.tensor([0])])  # shift right for teacher forcing
        return x, y

    def __len__(self):
        return len(self.samples)


class Multi30KDataset(Dataset):
    """
    Multi30K Dataset for English-German Translation
    Supports any tokenizer that implements BaseTokenizer interface
    """

    def __init__(
            self,
            split: str = "train",
            src_tokenizer: BaseTokenizer = None,
            tgt_tokenizer: BaseTokenizer = None,
            src_tokenizer_path: str = None,
            tgt_tokenizer_path: str = None,
            tokenizer_type: str = "bpe",
            max_len: int = 128,
    ):
        """
        Args:
            split: Dataset split ("train", "validation", "test")
            src_tokenizer: Pre-loaded source tokenizer (optional)
            tgt_tokenizer: Pre-loaded target tokenizer (optional)
            src_tokenizer_path: Path to source tokenizer file (if not pre-loaded)
            tgt_tokenizer_path: Path to target tokenizer file (if not pre-loaded)
            tokenizer_type: Type of tokenizer ("bpe", "wordpiece", "unigram", "simple")
            max_len: Maximum sequence length
        """
        self.split = split
        self.max_len = max_len

        # Load dataset
        print(f"Loading {split} dataset...")
        self.dataset = load_dataset("bentrevett/multi30k", split=split)

        # Load or use provided tokenizers
        print(f"Loading tokenizers...")

        if src_tokenizer is None and src_tokenizer_path is None and tokenizer_type is not None:
            src_tokenizer_path = f"tokenizer_en_{tokenizer_type}.json"

        if src_tokenizer is not None:
            self.src_tokenizer = src_tokenizer
        elif src_tokenizer_path is not None:
            self.src_tokenizer = self._load_tokenizer(src_tokenizer_path, tokenizer_type)
        else:
            raise ValueError("Must provide either src_tokenizer or src_tokenizer_path")

        if tgt_tokenizer is None and tgt_tokenizer_path is None and tokenizer_type is not None:
            tgt_tokenizer_path = f"tokenizer_de_{tokenizer_type}.json"

        if tgt_tokenizer is not None:
            self.tgt_tokenizer = tgt_tokenizer
        elif tgt_tokenizer_path is not None:
            self.tgt_tokenizer = self._load_tokenizer(tgt_tokenizer_path, tokenizer_type)
        else:
            raise ValueError("Must provide either tgt_tokenizer or tgt_tokenizer_path")

        # Get special token IDs (all tokenizers should have the same IDs)
        self.pad_id = self.src_tokenizer.pad_id
        self.sos_id = self.src_tokenizer.sos_id
        self.eos_id = self.src_tokenizer.eos_id
        self.unk_id = self.src_tokenizer.unk_id

        # Verify consistency
        self._verify_special_tokens()

        # Vocabulary sizes
        self.src_vocab_size = self.src_tokenizer.vocab_size
        self.tgt_vocab_size = self.tgt_tokenizer.vocab_size
        self.vocab_size = max(self.src_vocab_size, self.tgt_vocab_size)

        print(f"\nDataset Info:")
        print(f"  Split: {split}")
        print(f"  Examples: {len(self.dataset)}")
        print(f"  Source vocab: {self.src_vocab_size}")
        print(f"  Target vocab: {self.tgt_vocab_size}")
        print(f"  Max length: {max_len}")
        print(f"\nSpecial Tokens:")
        print(f"  PAD: {self.pad_id}")
        print(f"  UNK: {self.unk_id}")
        print(f"  SOS: {self.sos_id}")
        print(f"  EOS: {self.eos_id}")

    def _load_tokenizer(self, path: str, tokenizer_type: str) -> BaseTokenizer:
        """Load tokenizer based on type"""
        from tokenizer import (
            SimpleTokenizer,
            BPETokenizer,
            WordPieceTokenizer,
            UnigramTokenizer,
            CharacterTokenizer,
            SentencePieceStyleTokenizer,
            ByteLevelBPETokenizer,
        )

        tokenizer_classes = {
            "simple": SimpleTokenizer,
            "bpe": BPETokenizer,
            "wordpiece": WordPieceTokenizer,
            "unigram": UnigramTokenizer,
            "char": CharacterTokenizer,
            "sentencepiece": SentencePieceStyleTokenizer,
            "bytelevel": ByteLevelBPETokenizer,
        }

        if tokenizer_type not in tokenizer_classes:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

        TokenizerClass = tokenizer_classes[tokenizer_type]
        return TokenizerClass.load(path)

    def _verify_special_tokens(self):
        """Verify that source and target tokenizers have consistent special token IDs"""
        assert self.src_tokenizer.pad_id == self.tgt_tokenizer.pad_id, "PAD ID mismatch!"
        assert self.src_tokenizer.sos_id == self.tgt_tokenizer.sos_id, "SOS ID mismatch!"
        assert self.src_tokenizer.eos_id == self.tgt_tokenizer.eos_id, "EOS ID mismatch!"
        assert self.src_tokenizer.unk_id == self.tgt_tokenizer.unk_id, "UNK ID mismatch!"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Encode with padding
        src_ids = self.src_tokenizer.encode(
            item["en"],
            add_special_tokens=True,
            max_length=self.max_len,
            padding=True,
        )

        tgt_ids = self.tgt_tokenizer.encode(
            item["de"],
            add_special_tokens=True,
            max_length=self.max_len,
            padding=True,
        )

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_txt": item["en"],
            "tgt_txt": item["de"],
        }

    def decode_src(self, token_ids, skip_special_tokens=True):
        """Decode source language tokens"""
        return self.src_tokenizer.decode(token_ids, skip_special_tokens)

    def decode_tgt(self, token_ids, skip_special_tokens=True):
        """Decode target language tokens"""
        return self.tgt_tokenizer.decode(token_ids, skip_special_tokens)


def test_multi30k():
    dataset = Multi30KDataset(split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    for batch in dataloader:
        print()
        print("SRC TXT:", batch['src_txt'][0])
        print("TGT TXT:", batch['tgt_txt'][0])
        print("SRC IDS:", batch['src_ids'][0])
        print("TGT IDS:", batch['tgt_ids'][0])
        break
