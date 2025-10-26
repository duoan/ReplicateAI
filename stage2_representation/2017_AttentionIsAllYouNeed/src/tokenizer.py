import json
from pathlib import Path
from typing import Iterable, List, Union, Optional

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

class BPETokenizer:
    """Simple wrapper around tokenizers.Tokenizer with convenient properties"""
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    SOS_TOKEN = "[SOS]"
    EOS_TOKEN = "[EOS]"
    
    # Special token IDs
    PAD_ID = 0
    UNK_ID = 1
    SOS_ID = 2
    EOS_ID = 3
    
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    
    def __init__(self, tokenizer: Tokenizer):
        """
        Args:
            tokenizer: Trained tokenizers.Tokenizer object
        """
        self._tokenizer = tokenizer
        self._verify_special_tokens()
    
    def _verify_special_tokens(self):
        """Verify special token IDs are correct"""
        for token, expected_id in zip(self.SPECIAL_TOKENS, [self.PAD_ID, self.UNK_ID, self.SOS_ID, self.EOS_ID]):
            actual_id = self._tokenizer.token_to_id(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"Special token '{token}' has ID {actual_id}, expected {expected_id}"
                )
    
    @property
    def tokenizer(self) -> Tokenizer:
        """Get underlying tokenizers.Tokenizer object"""
        return self._tokenizer
    
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
    
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
    
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()
        tokens = [self._tokenizer.id_to_token(i) for i in ids]
        return [t for t in tokens if t not in self.SPECIAL_TOKENS]

    
    def save(self, path: str):
        """Save tokenizer"""
        self._tokenizer.save(path)
        
        # Save config
        config_path = Path(path).with_suffix(".config.json")
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad_token": self.PAD_TOKEN,
                "unk_token": self.UNK_TOKEN,
                "sos_token": self.SOS_TOKEN,
                "eos_token": self.EOS_TOKEN,
                "pad_id": self.PAD_ID,
                "unk_id": self.UNK_ID,
                "sos_id": self.SOS_ID,
                "eos_id": self.EOS_ID,
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Tokenizer saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from file"""
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer)
    
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """
        Train a new BPE tokenizer
        
        Args:
            texts: Iterable of training texts
            vocab_size: Vocabulary size
            min_frequency: Minimum token frequency
            
        Returns:
            BPETokenizer object
        """
        # Initialize BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token=cls.UNK_TOKEN))
        
        # Setup trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=cls.SPECIAL_TOKENS,
            min_frequency=min_frequency,
            show_progress=True,
        )
        
        # Setup normalizer and pre-tokenizer
        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        
        # Train
        tokenizer.train_from_iterator(texts, trainer=trainer)
        
        # Setup post-processor (auto add SOS/EOS)
        tokenizer.post_processor = TemplateProcessing(
            single=f"{cls.SOS_TOKEN} $A {cls.EOS_TOKEN}",
            special_tokens=[
                (cls.SOS_TOKEN, cls.SOS_ID),
                (cls.EOS_TOKEN, cls.EOS_ID),
            ],
        )
        
        # Enable padding
        tokenizer.enable_padding(
            pad_id=cls.PAD_ID,
            pad_token=cls.PAD_TOKEN,
        )
        
        # Enable truncation
        tokenizer.enable_truncation(max_length=512)
        
        return cls(tokenizer)
    
    def __len__(self) -> int:
        return self.vocab_size
    
    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"pad_id={self.pad_id}, unk_id={self.unk_id}, "
            f"sos_id={self.sos_id}, eos_id={self.eos_id})"
        )


def get_tokenizer(path: str) -> BPETokenizer:
    """
    Load a trained tokenizer from file
    
    Args:
        path: Path to tokenizer file
        
    Returns:
        BPETokenizer object
    """
    return BPETokenizer.load(path)


def evaluate_tokenizer(tokenizer: BPETokenizer, texts: Iterable[str], name: str = "", max_samples: int = 1000):
    """Evaluate tokenizer quality"""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {name} Tokenizer")
    print(f"{'=' * 60}")
    print(tokenizer)
    
    # Convert to list for multiple passes (limit to max_samples)
    text_list = []
    for i, text in enumerate(texts):
        if i >= max_samples:
            break
        text_list.append(text)
    
    if len(text_list) == 0:
        print("⚠️  No texts to evaluate!")
        return
    
    # Calculate UNK rate
    total_tokens = 0
    unk_tokens = 0
    
    for text in tqdm(text_list, desc="Calculating UNK rate"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(ids)
        unk_tokens += (ids == tokenizer.unk_id).sum().item()
    
    unk_rate = unk_tokens / total_tokens * 100 if total_tokens > 0 else 0
    print(f"UNK rate: {unk_rate:.2f}%")
    
    # Calculate average length
    lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in text_list]
    avg_len = sum(lengths) / len(lengths)
    print(f"Avg tokens/sentence: {avg_len:.2f}")
    
    # Test examples
    print("\nExample encodings:")
    for text in text_list[:3]:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  Original: {text}")
        print(f"  IDs: {ids.tolist()[:15]}...")
        print(f"  Decoded: {decoded}")
        print()


def text_iterator(dataset, lang_key):
    """Create an iterator for a specific language"""
    for item in dataset:
        yield item["translation"][lang_key]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BPE tokenizers")
    parser.add_argument("--dataset", type=str, default="wmt/wmt14")
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default=str(SCRIPT_DIR))
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training BPE Tokenizers")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.dataset, name="de-en", split="train")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Train English tokenizer
    print("\n" + "-" * 60)
    print("Training English tokenizer...")
    print("-" * 60)
    en_texts = text_iterator(dataset, "en")
    en_tokenizer = BPETokenizer.train(
        texts=en_texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
    )
    en_path = f"{args.save_dir}/tokenizer_en.json"
    en_tokenizer.save(en_path)
    
    # Evaluate English tokenizer (create new iterator)
    en_texts_eval = text_iterator(dataset, "en")
    evaluate_tokenizer(en_tokenizer, en_texts_eval, "English", max_samples=1000)
    
    # Train German tokenizer
    print("\n" + "-" * 60)
    print("Training German tokenizer...")
    print("-" * 60)
    de_texts = text_iterator(dataset, "de")
    de_tokenizer = BPETokenizer.train(
        texts=de_texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
    )
    de_path = f"{args.save_dir}/tokenizer_de.json"
    de_tokenizer.save(de_path)
    
    # Evaluate German tokenizer (create new iterator)
    de_texts_eval = text_iterator(dataset, "de")
    evaluate_tokenizer(de_tokenizer, de_texts_eval, "German", max_samples=1000)
    
    # Verify special token consistency
    print("\n" + "=" * 60)
    print("Special Token Verification")
    print("=" * 60)
    
    print(f"{'Token':<10} {'EN ID':<10} {'DE ID':<10} {'Status'}")
    print("-" * 60)
    for token_name in ["PAD", "UNK", "SOS", "EOS"]:
        en_id = getattr(en_tokenizer, f"{token_name.lower()}_id")
        de_id = getattr(de_tokenizer, f"{token_name.lower()}_id")
        status = "✓" if en_id == de_id else "✗"
        print(f"{token_name:<10} {en_id:<10} {de_id:<10} {status}")
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print("=" * 60)
    print(f"English tokenizer: {en_path}")
    print(f"German tokenizer: {de_path}")
