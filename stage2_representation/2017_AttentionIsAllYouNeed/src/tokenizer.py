# tokenizer.py
import json
from pathlib import Path
from typing import List, Union, Optional
from abc import ABC, abstractmethod

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tqdm import tqdm


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers
    Defines the common interface
    """

    # Define special tokens (meaningful strings)
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    SOS_TOKEN = "[SOS]"
    EOS_TOKEN = "[EOS]"

    # Force specific IDs for special tokens (ensure consistency)
    PAD_ID = 0
    UNK_ID = 1
    SOS_ID = 2
    EOS_ID = 3

    @classmethod
    def get_special_tokens(cls) -> List[str]:
        """Get special tokens list (in ID order)"""
        return [cls.PAD_TOKEN, cls.UNK_TOKEN, cls.SOS_TOKEN, cls.EOS_TOKEN]

    @abstractmethod
    def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
            max_length: Optional[int] = None,
            padding: bool = False,
    ) -> torch.Tensor:
        """Encode text to token IDs"""
        pass

    @abstractmethod
    def decode(
            self,
            token_ids: Union[List[int], torch.Tensor],
            skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text"""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save tokenizer"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """Load tokenizer"""
        pass

    @classmethod
    @abstractmethod
    def train(cls, texts: List[str], vocab_size: int, min_frequency: int):
        """Train tokenizer"""
        pass

    @abstractmethod
    def token_in_vocab(self, token: str) -> bool:
        """Check if a token is in vocabulary"""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size"""
        pass

    @property
    def pad_id(self) -> int:
        return self.PAD_ID

    @property
    def unk_id(self) -> int:
        return self.UNK_ID

    @property
    def sos_id(self) -> int:
        return self.SOS_ID

    @property
    def eos_id(self) -> int:
        return self.EOS_ID

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  special_tokens={{\n"
            f"    '{self.PAD_TOKEN}': {self.PAD_ID},\n"
            f"    '{self.UNK_TOKEN}': {self.UNK_ID},\n"
            f"    '{self.SOS_TOKEN}': {self.SOS_ID},\n"
            f"    '{self.EOS_TOKEN}': {self.EOS_ID},\n"
            f"  }}\n"
            f")"
        )


class SimpleTokenizer(BaseTokenizer):
    """
    Simple word-level tokenizer (baseline)
    Splits on whitespace, no subword tokenization
    Pure Python implementation
    """

    def __init__(self):
        """Initialize empty tokenizer. Use train() or load() to create."""
        self._word2id = {}
        self._id2word = {}

    @classmethod
    def train(
            cls,
            texts: List[str],
            vocab_size: int = 10000,
            min_frequency: int = 2,
    ) -> 'SimpleTokenizer':
        """Train a simple word-level tokenizer"""
        from collections import Counter

        print("Building word vocabulary...")

        # Count word frequencies
        word_counter = Counter()
        for text in tqdm(texts, desc="Counting words"):
            words = text.lower().split()
            word_counter.update(words)

        print(f"  Total unique words: {len(word_counter)}")

        # Get most common words that meet min_frequency
        special_tokens = cls.get_special_tokens()
        most_common = word_counter.most_common()
        vocab_words = [word for word, freq in most_common if freq >= min_frequency]

        # Limit to vocab_size
        vocab_words = vocab_words[:vocab_size - len(special_tokens)]

        print(f"  After min_freq={min_frequency}: {len(vocab_words)}")

        # Build vocabulary: special tokens + words
        vocab = special_tokens + vocab_words

        print(f"  Final vocab size: {len(vocab)}")

        # Create instance and set vocabulary
        instance = cls()
        instance._word2id = {word: idx for idx, word in enumerate(vocab)}
        instance._id2word = {idx: word for word, idx in instance._word2id.items()}

        return instance

    def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
            max_length: Optional[int] = None,
            padding: bool = False,
    ) -> torch.Tensor:
        """Encode text to token IDs"""
        words = text.lower().split()

        ids = []
        if add_special_tokens:
            ids.append(self.SOS_ID)

        for word in words:
            word_id = self._word2id.get(word, self.UNK_ID)
            ids.append(word_id)

        if add_special_tokens:
            ids.append(self.EOS_ID)

        # Truncate if needed
        if max_length is not None and len(ids) > max_length:
            if add_special_tokens:
                ids = ids[:max_length - 1] + [self.EOS_ID]
            else:
                ids = ids[:max_length]

        # Pad if needed
        if padding and max_length is not None:
            while len(ids) < max_length:
                ids.append(self.PAD_ID)

        return torch.tensor(ids, dtype=torch.long)

    def decode(
            self,
            token_ids: Union[List[int], torch.Tensor],
            skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        words = []
        special_ids = {self.PAD_ID, self.SOS_ID, self.EOS_ID} if skip_special_tokens else set()

        for token_id in token_ids:
            if token_id in special_ids:
                continue

            word = self._id2word.get(token_id, self.UNK_TOKEN)
            words.append(word)

        return ' '.join(words)

    def token_in_vocab(self, token: str) -> bool:
        """Check if a token is in vocabulary"""
        return token.lower() in self._word2id

    def save(self, path: str):
        """Save tokenizer"""
        data = {
            'tokenizer_type': self.__class__.__name__,
            'word2id': self._word2id,
            'special_tokens': {
                'pad_token': self.PAD_TOKEN,
                'unk_token': self.UNK_TOKEN,
                'sos_token': self.SOS_TOKEN,
                'eos_token': self.EOS_TOKEN,
                'pad_id': self.PAD_ID,
                'unk_id': self.UNK_ID,
                'sos_id': self.SOS_ID,
                'eos_id': self.EOS_ID,
            },
            'vocab_size': len(self._word2id),
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Tokenizer saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create instance and set vocabulary
        instance = cls()
        instance._word2id = data['word2id']
        instance._id2word = {int(v): k for k, v in instance._word2id.items()}

        return instance

    @property
    def vocab_size(self) -> int:
        """Vocabulary size"""
        return len(self._word2id)


class HuggingFaceTokenizer(BaseTokenizer):
    """
    Base class for tokenizers using HuggingFace tokenizers library
    Provides common functionality for BPE, WordPiece, Unigram, etc.
    """

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        """
        Args:
            tokenizer: Trained tokenizers.Tokenizer object
        """
        self._tokenizer = tokenizer

        if tokenizer is not None:
            self._verify_special_tokens()

    def _verify_special_tokens(self):
        """Verify special token IDs are correct"""
        expected = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.SOS_TOKEN: self.SOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
        }

        for token, expected_id in expected.items():
            actual_id = self._tokenizer.token_to_id(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"Special token '{token}' has ID {actual_id}, "
                    f"but expected {expected_id}"
                )

    def _setup_tokenizer(self, tokenizer: Tokenizer):
        """Setup post-processor, padding, and truncation"""
        # Set post-processor (auto add SOS/EOS)
        tokenizer.post_processor = TemplateProcessing(
            single=f"{self.SOS_TOKEN} $A {self.EOS_TOKEN}",
            special_tokens=[
                (self.SOS_TOKEN, self.SOS_ID),
                (self.EOS_TOKEN, self.EOS_ID),
            ],
        )

        # Enable padding
        tokenizer.enable_padding(
            pad_id=self.PAD_ID,
            pad_token=self.PAD_TOKEN,
        )

        # Enable truncation
        tokenizer.enable_truncation(max_length=512)

        return tokenizer

    def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
            max_length: Optional[int] = None,
            padding: bool = False,
    ) -> torch.Tensor:
        """Encode text to token IDs"""
        if max_length is not None:
            self._tokenizer.enable_truncation(max_length=max_length)

        if padding and max_length is not None:
            self._tokenizer.enable_padding(
                pad_id=self.PAD_ID,
                pad_token=self.PAD_TOKEN,
                length=max_length,
            )

        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return torch.tensor(encoding.ids, dtype=torch.long)

    def decode(
            self,
            token_ids: Union[List[int], torch.Tensor],
            skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def token_in_vocab(self, token: str) -> bool:
        """Check if a token is in vocabulary"""
        return self._tokenizer.token_to_id(token) is not None

    def save(self, path: str):
        """Save tokenizer"""
        # Save tokenizer
        self._tokenizer.save(path)

        # Save config
        config_path = Path(path).with_suffix('.config.json')
        config = {
            'tokenizer_type': self.__class__.__name__,
            'pad_token': self.PAD_TOKEN,
            'unk_token': self.UNK_TOKEN,
            'sos_token': self.SOS_TOKEN,
            'eos_token': self.EOS_TOKEN,
            'pad_id': self.PAD_ID,
            'unk_id': self.UNK_ID,
            'sos_id': self.SOS_ID,
            'eos_id': self.EOS_ID,
            'vocab_size': self.vocab_size,
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✓ Tokenizer saved to: {path}")
        print(f"✓ Config saved to: {config_path}")

    @classmethod
    def load(cls, path: str):
        """Load tokenizer"""
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size"""
        return self._tokenizer.get_vocab_size()


class BPETokenizer(HuggingFaceTokenizer):
    """BPE (Byte Pair Encoding) Tokenizer"""

    @classmethod
    def train(
            cls,
            texts: List[str],
            vocab_size: int = 10000,
            min_frequency: int = 2,
    ) -> 'BPETokenizer':
        """Train a new BPE tokenizer"""
        tokenizer = Tokenizer(BPE(unk_token=cls.UNK_TOKEN))

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=cls.get_special_tokens(),
            min_frequency=min_frequency,
            show_progress=True,
        )

        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(texts, trainer=trainer)

        instance = cls()
        instance._tokenizer = instance._setup_tokenizer(tokenizer)
        instance._verify_special_tokens()

        return instance


class WordPieceTokenizer(HuggingFaceTokenizer):
    """WordPiece Tokenizer (used by BERT)"""

    @classmethod
    def train(
            cls,
            texts: List[str],
            vocab_size: int = 10000,
            min_frequency: int = 2,
    ) -> 'WordPieceTokenizer':
        """Train a new WordPiece tokenizer"""
        tokenizer = Tokenizer(WordPiece(unk_token=cls.UNK_TOKEN))

        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=cls.get_special_tokens(),
            min_frequency=min_frequency,
            show_progress=True,
        )

        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(texts, trainer=trainer)

        instance = cls()
        instance._tokenizer = instance._setup_tokenizer(tokenizer)
        instance._verify_special_tokens()

        return instance


class UnigramTokenizer(HuggingFaceTokenizer):
    """Unigram Tokenizer (used by SentencePiece)"""

    @classmethod
    def train(
            cls,
            texts: List[str],
            vocab_size: int = 10000,
            min_frequency: int = 2,
    ) -> 'UnigramTokenizer':
        """Train a new Unigram tokenizer"""
        tokenizer = Tokenizer(Unigram())

        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=cls.get_special_tokens(),
            unk_token=cls.UNK_TOKEN,
            show_progress=True,
        )

        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(texts, trainer=trainer)

        instance = cls()
        instance._tokenizer = instance._setup_tokenizer(tokenizer)
        instance._verify_special_tokens()

        return instance


class CharacterTokenizer(HuggingFaceTokenizer):
    """Character-level Tokenizer (baseline)"""

    @classmethod
    def train(
            cls,
            texts: List[str],
            vocab_size: int = 10000,
            min_frequency: int = 1,
    ) -> 'CharacterTokenizer':
        """Train a character-level tokenizer"""
        char_freq = {}
        for text in texts:
            for char in text.lower():
                char_freq[char] = char_freq.get(char, 0) + 1

        vocab = {char: freq for char, freq in char_freq.items() if freq >= min_frequency}

        special_tokens = cls.get_special_tokens()
        vocab_list = special_tokens + sorted(vocab.keys(), key=lambda x: vocab[x], reverse=True)

        if len(vocab_list) > vocab_size:
            vocab_list = vocab_list[:vocab_size]

        tokenizer = Tokenizer(BPE(unk_token=cls.UNK_TOKEN))

        trainer = BpeTrainer(
            vocab_size=len(vocab_list),
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=True,
        )

        tokenizer.normalizer = Sequence([Lowercase()])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(texts, trainer=trainer)

        instance = cls()
        instance._tokenizer = instance._setup_tokenizer(tokenizer)
        instance._verify_special_tokens()

        return instance


class SentencePieceStyleTokenizer(HuggingFaceTokenizer):
    """SentencePiece-style tokenizer"""

    @classmethod
    def train(
            cls,
            texts: List[str],
            vocab_size: int = 10000,
            min_frequency: int = 2,
    ) -> 'SentencePieceStyleTokenizer':
        """Train a SentencePiece-style tokenizer"""
        tokenizer = Tokenizer(Unigram())

        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=cls.get_special_tokens(),
            unk_token=cls.UNK_TOKEN,
            show_progress=True,
        )

        tokenizer.normalizer = None
        tokenizer.pre_tokenizer = None
        tokenizer.train_from_iterator(texts, trainer=trainer)

        instance = cls()
        instance._tokenizer = instance._setup_tokenizer(tokenizer)
        instance._verify_special_tokens()

        return instance


class ByteLevelBPETokenizer(HuggingFaceTokenizer):
    """Byte-level BPE tokenizer (like GPT-2)"""

    @classmethod
    def train(
            cls,
            texts: List[str],
            vocab_size: int = 10000,
            min_frequency: int = 2,
    ) -> 'ByteLevelBPETokenizer':
        """Train a byte-level BPE tokenizer"""
        tokenizer = Tokenizer(BPE(unk_token=cls.UNK_TOKEN))

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=cls.get_special_tokens(),
            min_frequency=min_frequency,
            show_progress=True,
        )

        tokenizer.normalizer = None
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.train_from_iterator(texts, trainer=trainer)

        instance = cls()
        instance._tokenizer = instance._setup_tokenizer(tokenizer)
        instance._verify_special_tokens()

        return instance


# ============================================================================
# Utility Functions (now work with unified interface)
# ============================================================================

def check_word_level_coverage(tokenizer: BaseTokenizer, texts: List[str]):
    """Check word-level coverage"""
    unique_tokens = set()
    for text in texts:
        tokens = text.lower().split()
        unique_tokens.update(tokens)

    covered = 0
    for token in unique_tokens:
        if tokenizer.token_in_vocab(token):
            covered += 1

    coverage = covered / len(unique_tokens) * 100
    print(f"Vocab coverage: {coverage:.2f}%")
    print(f"Unique tokens: {len(unique_tokens)}")


def check_unk_rate(tokenizer: BaseTokenizer, texts: List[str]):
    """Check UNK rate"""
    total_tokens = 0
    unk_tokens = 0

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(ids)
        unk_tokens += (ids == tokenizer.unk_id).sum().item()

    unk_rate = unk_tokens / total_tokens * 100
    print(f"UNK rate: {unk_rate:.2f}%")
    return unk_rate


def check_avg_length(tokenizer: BaseTokenizer, texts: List[str]):
    """Check average token length"""
    lengths = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(ids))

    avg_len = sum(lengths) / len(lengths)
    print(f"Average tokens per sentence: {avg_len:.2f}")


def analyze_subword_stats(tokenizer: BaseTokenizer, texts: List[str], name=""):
    """Analyze subword tokenization statistics"""
    word_counts = []
    subword_counts = []

    for text in texts[:1000]:
        words = text.lower().split()
        ids = tokenizer.encode(text, add_special_tokens=False)

        word_counts.append(len(words))
        subword_counts.append(len(ids))

    avg_words = sum(word_counts) / len(word_counts)
    avg_subwords = sum(subword_counts) / len(subword_counts)
    ratio = avg_subwords / avg_words

    print(f"\n{name} Subword Analysis:")
    print(f"  Avg words/sentence: {avg_words:.2f}")
    print(f"  Avg subwords/sentence: {avg_subwords:.2f}")
    print(f"  Subword/Word ratio: {ratio:.2f}")

    if isinstance(tokenizer, SimpleTokenizer):
        print(f"  → {'Good (word-level)' if 0.95 <= ratio <= 1.05 else 'Unexpected for word-level'}")
    else:
        print(f"  → {'Good' if ratio < 1.5 else 'Too fragmented'}")


def check_unk_rate_on_splits(en_tokenizer: BaseTokenizer, de_tokenizer: BaseTokenizer, dataset):
    """Check UNK rate on different splits"""
    print("\n" + "=" * 80)
    print("UNK Rate Analysis")
    print("=" * 80)

    for split in ["train", "validation", "test"]:
        en_total, en_unk = 0, 0
        de_total, de_unk = 0, 0

        for item in dataset[split]:
            en_ids = en_tokenizer.encode(item["en"], add_special_tokens=False)
            en_total += len(en_ids)
            en_unk += (en_ids == en_tokenizer.unk_id).sum().item()

            de_ids = de_tokenizer.encode(item["de"], add_special_tokens=False)
            de_total += len(de_ids)
            de_unk += (de_ids == de_tokenizer.unk_id).sum().item()

        en_unk_rate = en_unk / en_total * 100 if en_total > 0 else 0
        de_unk_rate = de_unk / de_total * 100 if de_total > 0 else 0

        print(f"\n{split.upper()}:")
        print(f"  EN UNK rate: {en_unk_rate:.2f}%")
        print(f"  DE UNK rate: {de_unk_rate:.2f}%")


def train_tokenizers(
        dataset_name: str = "bentrevett/multi30k",
        tokenizer_type: str = "bpe",
        vocab_size: int = 5000,
        min_frequency: int = 3,
        save_dir: str = ".",
):
    """Train English and German tokenizers"""

    print("=" * 80)
    print(f"Training {tokenizer_type.upper()} Tokenizers")
    print("=" * 80)

    # Select tokenizer class
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

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_name)

    # Extract texts
    print("Extracting texts...")
    en_texts = []
    de_texts = []

    for split in ["train"]:
        for item in tqdm(dataset[split], desc=f"Processing {split}"):
            en_texts.append(item["en"])
            de_texts.append(item["de"])

    print(f"  English texts: {len(en_texts)}")
    print(f"  German texts: {len(de_texts)}")

    # Train English tokenizer
    print("\n" + "-" * 80)
    print("Training English Tokenizer")
    print("-" * 80)
    en_tokenizer = TokenizerClass.train(
        texts=en_texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    en_save_path = f"{save_dir}/tokenizer_en_{tokenizer_type}.json"
    en_tokenizer.save(en_save_path)
    print(en_tokenizer)
    check_word_level_coverage(en_tokenizer, en_texts)
    check_unk_rate(en_tokenizer, en_texts)
    check_avg_length(en_tokenizer, en_texts)
    analyze_subword_stats(en_tokenizer, en_texts)

    # Train German tokenizer
    print("\n" + "-" * 80)
    print("Training German Tokenizer")
    print("-" * 80)
    de_tokenizer = TokenizerClass.train(
        texts=de_texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    de_save_path = f"{save_dir}/tokenizer_de_{tokenizer_type}.json"
    de_tokenizer.save(de_save_path)
    print(de_tokenizer)

    check_word_level_coverage(de_tokenizer, de_texts)
    check_unk_rate(de_tokenizer, de_texts)
    check_avg_length(de_tokenizer, de_texts)
    analyze_subword_stats(de_tokenizer, de_texts)

    check_unk_rate_on_splits(en_tokenizer, de_tokenizer, dataset)

    # Verify special token consistency
    print("\n" + "=" * 80)
    print("Verifying Special Token Consistency")
    print("=" * 80)

    special_tokens = [
        ("PAD", en_tokenizer.pad_id, de_tokenizer.pad_id),
        ("UNK", en_tokenizer.unk_id, de_tokenizer.unk_id),
        ("SOS", en_tokenizer.sos_id, de_tokenizer.sos_id),
        ("EOS", en_tokenizer.eos_id, de_tokenizer.eos_id),
    ]

    all_match = True
    print(f"{'Token':<10} {'EN ID':<10} {'DE ID':<10} {'Status'}")
    print("-" * 80)
    for token_name, en_id, de_id in special_tokens:
        match = "✓" if en_id == de_id else "✗"
        print(f"{token_name:<10} {en_id:<10} {de_id:<10} {match}")
        if en_id != de_id:
            all_match = False

    if all_match:
        print("\n✓ All special token IDs match!")
    else:
        print("\n✗ ERROR: Special token IDs don't match!")
        raise ValueError("Special token IDs must be consistent!")

    # Test examples
    print("\n" + "=" * 80)
    print("Testing Tokenizers")
    print("=" * 80)

    test_en = "A man in an orange hat starring at something."
    test_de = "Ein Mann mit einem orangefarbenen Hut starrt auf etwas."

    print(f"\nEnglish:")
    print(f"  Original: {test_en}")
    en_ids = en_tokenizer.encode(test_en)
    print(f"  IDs: {en_ids.tolist()}")
    print(f"  Decoded: {en_tokenizer.decode(en_ids)}")

    print(f"\nGerman:")
    print(f"  Original: {test_de}")
    de_ids = de_tokenizer.encode(test_de)
    print(f"  IDs: {de_ids.tolist()}")
    print(f"  Decoded: {de_tokenizer.decode(de_ids)}")

    print("\n" + "=" * 80)
    print("✓ Tokenizers trained and saved successfully!")
    print("=" * 80)

    return en_tokenizer, de_tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train tokenizers for machine translation")
    parser.add_argument(
        "--type",
        type=str,
        default="bpe",
        choices=["bpe", "wordpiece", "unigram", "char", "simple", "sentencepiece", "bytelevel"],
        help="Tokenizer type"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=5000,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help="Minimum frequency"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=".",
        help="Save directory"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train all tokenizer types for comparison"
    )

    args = parser.parse_args()

    if args.compare:
        print("\n" + "=" * 80)
        print("TRAINING ALL TOKENIZER TYPES FOR COMPARISON")
        print("=" * 80)

        tokenizer_types = ["simple", "bpe", "wordpiece", "unigram"]
        results = {}

        for tok_type in tokenizer_types:
            print(f"\n{'=' * 80}")
            print(f"Training {tok_type.upper()}")
            print(f"{'=' * 80}")

            try:
                train_tokenizers(
                    tokenizer_type=tok_type,
                    vocab_size=args.vocab_size,
                    min_frequency=args.min_freq,
                    save_dir=args.save_dir,
                )
                results[tok_type] = "✓ Success"
            except Exception as e:
                results[tok_type] = f"✗ Failed: {str(e)}"

        # Summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        for tok_type, status in results.items():
            print(f"{tok_type:15s}: {status}")

        # Comparison
        print("\n" + "=" * 80)
        print("TOKENIZER COMPARISON")
        print("=" * 80)

        test_text = "A young girl in a pink dress is climbing a set of stairs."

        print(f"\nTest sentence: {test_text}\n")
        print(f"{'Type':<15} {'Tokens':<8} {'Vocab':<8} {'Example IDs'}")
        print("-" * 80)

        tokenizer_classes = {
            "simple": SimpleTokenizer,
            "bpe": BPETokenizer,
            "wordpiece": WordPieceTokenizer,
            "unigram": UnigramTokenizer,
        }

        for tok_type in tokenizer_types:
            try:
                TokenizerClass = tokenizer_classes[tok_type]
                tokenizer = TokenizerClass.load(f"{args.save_dir}/tokenizer_en_{tok_type}.json")
                ids = tokenizer.encode(test_text, add_special_tokens=True)

                print(f"{tok_type:<15} {len(ids):<8} {tokenizer.vocab_size:<8} {ids[:10].tolist()}")
            except Exception as e:
                print(f"{tok_type:<15} {'N/A':<8} {'N/A':<8} Error: {str(e)}")

    else:
        train_tokenizers(
            tokenizer_type=args.type,
            vocab_size=args.vocab_size,
            min_frequency=args.min_freq,
            save_dir=args.save_dir,
        )
