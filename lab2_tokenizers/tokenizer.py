import json
import re
from collections import Counter

from transformers import PreTrainedTokenizer


class WhitespaceTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab=None,
        unk_token="<UNK>",
        pad_token="<PAD>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        **kwargs,
    ):
        self.vocab = vocab or {}
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

    def encode(self, text, add_special_tokens=False):
        tokens = self._tokenize(text)
        ids = [self._convert_token_to_id(tok) for tok in tokens]
        if add_special_tokens:
            ids = [self.vocab[self.bos_token]] + ids + [self.vocab[self.eos_token]]
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        tokens = [self._convert_id_to_token(i) for i in token_ids]
        if skip_special_tokens:
            tokens = [
                t
                for t in tokens
                if t
                not in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
            ]
        return self.convert_tokens_to_string(tokens)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def all_special_tokens(self):
        return [self.unk_token]

    @property
    def all_special_ids(self):
        return [self.vocab[self.unk_token]]

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text: str):
        words = text.split()
        tokens = []
        for word in words:
            tokens.extend(re.findall(r'\w+|[.,!?;:()"\']', word))
        return tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def build_vocab(self, texts, vocab_size=50000):
        counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)

        most_common = [t for t, _ in counter.most_common(vocab_size - 4)]

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

        vocab = {}

        for i, tok in enumerate(special_tokens):
            vocab[tok] = i

        for i, tok in enumerate(most_common, start=len(special_tokens)):
            vocab[tok] = i

        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}

    def save_pretrained(self, save_directory):
        vocab_file = f"{save_directory}/vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"Saved vocabulary to {vocab_file}")

    @classmethod
    def from_pretrained(cls, load_directory):
        vocab_file = f"{load_directory}/vocab.json"
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"Loaded vocabulary from {vocab_file}")
        return cls(vocab=vocab)
