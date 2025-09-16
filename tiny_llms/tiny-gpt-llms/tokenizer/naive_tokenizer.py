import os
from typing import Dict, List, Optional
import torch


class NaiveTokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, vocab: Optional[List[str]] = None):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._add_special_tokens()
        if vocab:
            for char in vocab:
                self.add_token(char)

    def _add_special_tokens(self):
        for token in [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def build_vocab_from_text(self, text: str) -> None:
        for char in sorted(set(text)):
            self.add_token(char)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.token_to_id[self.BOS_TOKEN])
        for char in text:
            tokens.append(self.token_to_id.get(char, self.token_to_id[self.UNK_TOKEN]))
        if add_special_tokens:
            tokens.append(self.token_to_id[self.EOS_TOKEN])
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        special_tokens = {
            self.token_to_id[self.PAD_TOKEN],
            self.token_to_id[self.UNK_TOKEN],
            self.token_to_id[self.BOS_TOKEN],
            self.token_to_id[self.EOS_TOKEN],
        }
        return "".join(
            self.id_to_token.get(t, self.UNK_TOKEN)
            for t in token_ids
            if not skip_special_tokens or t not in special_tokens
        )

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        batch_tokens = [self.encode(text, add_special_tokens) for text in texts]
        if max_length is None:
            max_length = max(len(tokens) for tokens in batch_tokens)
        if padding:
            pad_id = self.token_to_id[self.PAD_TOKEN]
            batch_tokens = [
                tokens[:max_length] + [pad_id] * max(0, max_length - len(tokens))
                for tokens in batch_tokens
            ]
        return torch.tensor(batch_tokens)

    def save(self, filepath: str) -> None:
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
        }
        torch.save(vocab_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "NaiveTokenizer":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")
        vocab_data = torch.load(filepath)
        tokenizer = cls()
        tokenizer.token_to_id = vocab_data["token_to_id"]
        tokenizer.id_to_token = {
            int(k): v for k, v in vocab_data["id_to_token"].items()
        }
        return tokenizer

    def __len__(self) -> int:
        return len(self.token_to_id)
