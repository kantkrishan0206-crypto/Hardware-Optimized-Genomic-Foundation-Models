from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

DNA_ALPHABET: tuple[str, ...] = ("A", "C", "G", "T", "N")
DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<unk>",
    "<mask>",
    "<cls>",
    "<sep>",
    "<chr>",
    "<var>",
    "<ref>",
    "<alt>",
    "<bos>",
    "<eos>",
    "<chr_start>",
    "<chr_end>",
)

TokenizationStrategy = Literal["nucleotide", "kmer", "bpe"]


@dataclass(frozen=True)
class BatchEncoding:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]


class GenomicTokenizer:
    """Tokenizer for nucleotide, k-mer, and simple genomic BPE workflows."""

    def __init__(
        self,
        strategy: TokenizationStrategy = "nucleotide",
        k: int = 3,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        if strategy not in {"nucleotide", "kmer", "bpe"}:
            raise ValueError(f"Unsupported tokenization strategy: {strategy}")
        if k < 1:
            raise ValueError("k must be positive for k-mer tokenization.")
        self.strategy = strategy
        self.k = k
        self.special_tokens = special_tokens
        self.merges = merges or []
        self.vocab = vocab or self._build_default_vocab(strategy, k, special_tokens)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.pad_token_id = self.vocab["<pad>"]
        self.unk_token_id = self.vocab["<unk>"]
        self.mask_token_id = self.vocab["<mask>"]
        self.cls_token_id = self.vocab["<cls>"]
        self.sep_token_id = self.vocab["<sep>"]
        self.bos_token_id = self.vocab["<bos>"]
        self.eos_token_id = self.vocab["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _build_default_vocab(
        strategy: TokenizationStrategy,
        k: int,
        special_tokens: tuple[str, ...],
    ) -> dict[str, int]:
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        if strategy == "nucleotide":
            symbols = DNA_ALPHABET
        elif strategy == "kmer":
            symbols = tuple("".join(parts) for parts in product(DNA_ALPHABET, repeat=k))
        else:
            symbols = DNA_ALPHABET
        for symbol in symbols:
            if symbol not in vocab:
                vocab[symbol] = len(vocab)
        return vocab

    @staticmethod
    def normalize(sequence: str) -> str:
        normalized = "".join(sequence.split()).upper()
        invalid = sorted(set(normalized) - set(DNA_ALPHABET))
        if invalid:
            raise ValueError(f"Unsupported genomic symbols: {', '.join(invalid)}")
        return normalized

    def tokenize(self, sequence: str) -> list[str]:
        sequence = self.normalize(sequence)
        if self.strategy == "nucleotide":
            return list(sequence)
        if self.strategy == "kmer":
            if len(sequence) < self.k:
                return [sequence] if sequence else []
            return [sequence[index : index + self.k] for index in range(len(sequence) - self.k + 1)]
        return self._bpe_tokenize(sequence)

    def _bpe_tokenize(self, sequence: str) -> list[str]:
        tokens = list(sequence)
        merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        while len(tokens) > 1:
            pairs = [(tokens[index], tokens[index + 1]) for index in range(len(tokens) - 1)]
            ranked = [(merge_ranks[pair], pair) for pair in pairs if pair in merge_ranks]
            if not ranked:
                break
            _, best_pair = min(ranked)
            merged: list[str] = []
            index = 0
            while index < len(tokens):
                if (
                    index < len(tokens) - 1
                    and tokens[index] == best_pair[0]
                    and tokens[index + 1] == best_pair[1]
                ):
                    merged.append(tokens[index] + tokens[index + 1])
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            tokens = merged
        return tokens

    def encode(
        self,
        sequence: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation: bool = False,
    ) -> list[int]:
        tokens = self.tokenize(sequence)
        if add_special_tokens:
            tokens = ["<cls>", *tokens, "<sep>"]
        if max_length is not None and len(tokens) > max_length:
            if not truncation:
                raise ValueError("Encoded sequence is longer than max_length and truncation is off.")
            tokens = tokens[:max_length]
            if add_special_tokens:
                tokens[-1] = "<sep>"
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = [self.id_to_token.get(token_id, "<unk>") for token_id in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        return "".join(tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def id_to_token_value(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, "<unk>")

    def mask_tokens(
        self,
        token_ids: list[int],
        mask_probability: float = 0.15,
        seed: int = 13,
    ) -> tuple[list[int], list[int]]:
        import random

        rng = random.Random(seed)
        masked: list[int] = []
        labels: list[int] = []
        protected = {self.pad_token_id, self.cls_token_id, self.sep_token_id}
        for token_id in token_ids:
            if token_id in protected or rng.random() > mask_probability:
                masked.append(token_id)
                labels.append(-100)
            else:
                masked.append(self.mask_token_id)
                labels.append(token_id)
        return masked, labels

    def batch_encode(
        self,
        sequences: list[str],
        max_length: int,
        truncation: bool = True,
        padding: bool = True,
    ) -> BatchEncoding:
        encoded = [
            self.encode(
                sequence,
                add_special_tokens=True,
                max_length=max_length,
                truncation=truncation,
            )
            for sequence in sequences
        ]
        if padding:
            padded: list[list[int]] = []
            masks: list[list[int]] = []
            for row in encoded:
                pad_count = max_length - len(row)
                if pad_count < 0:
                    raise ValueError("Encoded row exceeds max_length.")
                padded.append(row + [self.pad_token_id] * pad_count)
                masks.append([1] * len(row) + [0] * pad_count)
            return BatchEncoding(input_ids=padded, attention_mask=masks)
        return BatchEncoding(input_ids=encoded, attention_mask=[[1] * len(row) for row in encoded])

    def save_pretrained(self, output_dir: str | Path) -> Path:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy": self.strategy,
            "k": self.k,
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": list(self.special_tokens),
        }
        target = path / "tokenizer.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.save_huggingface_compat(path)
        return target

    def save_huggingface_compat(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        tokenizer_config = {
            "model_type": "hogfm-genomic-tokenizer",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "cls_token": "<cls>",
            "sep_token": "<sep>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "model_max_length": int(1e6),
            "created_at": int(time.time()),
        }
        special_tokens_map = {
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "cls_token": "<cls>",
            "sep_token": "<sep>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "additional_special_tokens": ["<chr>", "<var>", "<ref>", "<alt>", "<chr_start>"],
        }
        (path / "vocab.json").write_text(json.dumps(self.vocab, indent=2), encoding="utf-8")
        (path / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, indent=2),
            encoding="utf-8",
        )
        (path / "special_tokens_map.json").write_text(
            json.dumps(special_tokens_map, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> GenomicTokenizer:
        tokenizer_path = Path(path)
        if tokenizer_path.is_dir():
            tokenizer_path = tokenizer_path / "tokenizer.json"
        payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        merges = [tuple(item) for item in payload.get("merges", [])]
        return cls(
            strategy=payload["strategy"],
            k=int(payload["k"]),
            vocab={token: int(idx) for token, idx in payload["vocab"].items()},
            merges=merges,
            special_tokens=tuple(payload.get("special_tokens", DEFAULT_SPECIAL_TOKENS)),
        )


def train_bpe_tokenizer(
    sequences: list[str],
    vocab_size: int = 256,
    min_frequency: int = 2,
) -> GenomicTokenizer:
    if vocab_size <= len(DEFAULT_SPECIAL_TOKENS) + len(DNA_ALPHABET):
        raise ValueError("vocab_size must leave room for base symbols and merges.")

    tokenized = [list(GenomicTokenizer.normalize(sequence)) for sequence in sequences]
    vocab = GenomicTokenizer._build_default_vocab("bpe", 1, DEFAULT_SPECIAL_TOKENS)
    merges: list[tuple[str, str]] = []

    while len(vocab) < vocab_size:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for tokens in tokenized:
            pair_counts.update((tokens[index], tokens[index + 1]) for index in range(len(tokens) - 1))
        if not pair_counts:
            break
        best_pair, frequency = pair_counts.most_common(1)[0]
        if frequency < min_frequency:
            break
        merged_token = "".join(best_pair)
        if merged_token not in vocab:
            vocab[merged_token] = len(vocab)
            merges.append(best_pair)
        next_tokenized: list[list[str]] = []
        for tokens in tokenized:
            merged: list[str] = []
            index = 0
            while index < len(tokens):
                if (
                    index < len(tokens) - 1
                    and tokens[index] == best_pair[0]
                    and tokens[index + 1] == best_pair[1]
                ):
                    merged.append(merged_token)
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            next_tokenized.append(merged)
        tokenized = next_tokenized

    return GenomicTokenizer(strategy="bpe", vocab=vocab, merges=merges)


def benchmark_tokenizer(
    tokenizer: GenomicTokenizer,
    sequences: list[str],
    max_length: int = 4096,
) -> dict[str, float | int | str]:
    started = time.perf_counter()
    encoded = tokenizer.batch_encode(sequences, max_length=max_length)
    elapsed = time.perf_counter() - started
    total_bases = sum(len(sequence) for sequence in sequences)
    total_tokens = sum(sum(mask) for mask in encoded.attention_mask)
    return {
        "strategy": tokenizer.strategy,
        "k": tokenizer.k,
        "sequences": len(sequences),
        "bases": total_bases,
        "tokens": total_tokens,
        "seconds": elapsed,
        "bases_per_second": total_bases / max(elapsed, 1e-12),
        "tokens_per_second": total_tokens / max(elapsed, 1e-12),
    }
