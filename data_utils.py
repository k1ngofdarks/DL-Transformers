from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class Tokenizer:
    def __init__(self, max_size: int = 20000):
        self.max_size = max_size
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.word2id: dict[str, int] = {}
        self.id2word: dict[int, str] = {}

    def build_vocab(self, sentences: list[str]) -> None:
        counter: Counter[str] = Counter()
        for sentence in sentences:
            counter.update(sentence.split())

        words = [word for word, count in counter.items()]
        words = sorted(words, key=lambda word: counter[word], reverse=True)[: self.max_size]

        vocab = self.special_tokens + words
        self.word2id = {word: idx for idx, word in enumerate(vocab)}
        self.id2word = {idx: word for word, idx in self.word2id.items()}

    def encode(self, sentence: str) -> list[int]:
        ids = [self.word2id.get(word, self.word2id["<unk>"]) for word in sentence.split()]
        return [self.word2id["<bos>"]] + ids + [self.word2id["<eos>"]]

    def decode(self, ids: list[int]) -> str:
        words: list[str] = []
        for idx in ids:
            token = self.id2word.get(idx, "<unk>")
            if token in {"<bos>", "<pad>"}:
                continue
            if token == "<eos>":
                break
            words.append(token)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2id)


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_sentences: list[str],
        tgt_sentences: list[str],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len: int = 80,
    ):
        pairs: list[tuple[list[int], list[int]]] = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_ids = src_tokenizer.encode(src)[:max_len]
            tgt_ids = tgt_tokenizer.encode(tgt)[:max_len]
            pairs.append((src_ids, tgt_ids))

        pairs.sort(key=lambda sample: len(sample[0]))
        self.src_ids = [torch.tensor(src_ids) for src_ids, _ in pairs]
        self.tgt_ids = [torch.tensor(tgt_ids) for _, tgt_ids in pairs]

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int):
        return self.src_ids[idx], self.tgt_ids[idx]


def collate_fn(batch, pad_id_src: int, pad_id_tgt: int):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id_src)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id_tgt)
    return src_padded, tgt_padded


def load_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip().split("\n")


def load_data_splits() -> tuple[list[str], list[str], list[str], list[str], list[str], str]:
    candidate_folders = [Path("DL_Transformers/bhw2-data/data"), Path("bhw2-data/data")]
    data_folder = next((folder for folder in candidate_folders if folder.exists()), None)
    if data_folder is None:
        raise FileNotFoundError("Could not find bhw2-data/data folder")

    train_de = load_file(str(data_folder / "train.de-en.de"))
    train_en = load_file(str(data_folder / "train.de-en.en"))
    val_de = load_file(str(data_folder / "val.de-en.de"))
    val_en = load_file(str(data_folder / "val.de-en.en"))
    test_de = load_file(str(data_folder / "test1.de-en.de"))

    return train_de, train_en, val_de, val_en, test_de, str(data_folder)


def create_dataloaders(
    train_de: list[str],
    train_en: list[str],
    val_de: list[str],
    val_en: list[str],
    batch_size: int = 128,
    max_len: int = 80,
):
    de_tokenizer = Tokenizer(max_size=20000)
    en_tokenizer = Tokenizer(max_size=20000)

    de_tokenizer.build_vocab(train_de)
    en_tokenizer.build_vocab(train_en)

    train_dataset = TranslationDataset(train_de, train_en, de_tokenizer, en_tokenizer, max_len=max_len)
    val_dataset = TranslationDataset(val_de, val_en, de_tokenizer, en_tokenizer, max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            pad_id_src=de_tokenizer.word2id["<pad>"],
            pad_id_tgt=en_tokenizer.word2id["<pad>"],
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch,
            pad_id_src=de_tokenizer.word2id["<pad>"],
            pad_id_tgt=en_tokenizer.word2id["<pad>"],
        ),
    )

    return de_tokenizer, en_tokenizer, train_loader, val_loader
