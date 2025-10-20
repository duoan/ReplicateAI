import torch


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


class Multi30KDataset(torch.utils.data.Dataset):

    def __init__(self, split="train", max_len=128, tokenizer_name="Helsinki-NLP/opus-mt-en-de"):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.max_len = max_len
        self.pad_id = self._tokenizer.pad_token_id  # 58100
        self.eos_id = self._tokenizer.eos_token_id  # 0
        self.decoder_start_id = getattr(self._tokenizer, "decoder_start_token_id", None)
        if self.decoder_start_id is None:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(tokenizer_name)
            self.decoder_start_id = getattr(cfg, "decoder_start_token_id", None)

        self.dataset = (
            load_dataset("bentrevett/multi30k", split=split)
            .map(self._tokenize_batch, batched=True, batch_size=1000)
        )

    def _tokenize_batch(self, batch):
        src = self._tokenizer(
            batch["en"], padding="max_length", truncation=True, max_length=self.max_len
        )
        tgt = self._tokenizer(
            batch["de"], padding="max_length", truncation=True, max_length=self.max_len
        )

        tgt_ids = []
        tgt_mask = []
        for ids, mask in zip(tgt["input_ids"], tgt["attention_mask"]):
            if self.eos_id not in ids:
                last_real = max(i for i, t in enumerate(ids) if t != self.pad_id)
                ids[last_real] = self.eos_id

            new = [self.decoder_start_id] + ids
            if len(new) > self.max_len:
                if new[-1] == self.pad_id:
                    new = new[:-1]
                else:
                    new = new[:-1]
                    if new[-1] != self.eos_id:
                        new[-1] = self.eos_id

            m = [1 if tok != self.pad_id else 0 for tok in new]
            tgt_ids.append(new)
            tgt_mask.append(m)

        return {
            "src": src["input_ids"],
            "src_mask": src["attention_mask"],
            "tgt": tgt_ids,
            "tgt_mask": tgt_mask,
        }

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "src": torch.tensor(item["src"], dtype=torch.long),
            "tgt": torch.tensor(item["tgt"], dtype=torch.long),
            "src_mask": (torch.tensor(item["src"]) != self.pad_id).long(),
            "tgt_mask": (torch.tensor(item["tgt"]) != self.pad_id).long(),
            "src_text": item["en"],
            "tgt_text": item["de"],
        }

    def __len__(self):
        return len(self.dataset)


def test_multi30k():
    dataset = Multi30KDataset(split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    for batch in dataloader:
        print()
        print("SRC TEXT:", batch['src_text'][0])
        print("TGT TEXT:", batch['tgt_text'][0])
        print("SRC MASK:", batch['src_mask'][0])
        print("TGT MASK:", batch['tgt_mask'][0])
        print("SRC TOKENS:", batch['src'][0])
        print("TGT TOKENS:", batch['tgt'][0])
        break
