from __future__ import annotations
# Create a vocabulary wrapper
from typing import Dict


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self) -> None:
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.idx = 0

    def add_word(self, word: str) -> None:
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word: str) -> int:
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.word2idx)
