

Docs in train: 6453
Docs in eval: 166

### Whitespace sets:

- train: 44 551 053 tokens
- eval: 1 999 054 tokens

### Sentencepiece sets:

- train: 55 392 422 tokens
- eval: 2 490 498 tokens

### Gpt2-pl sets:

- train: 62 935 237
- eval: 2 837 691


Trening 

5 - epok na całym datasecie

Model:
    tokenizer_path = "tokenizers/gpt2-pl"
    max_new_tokens = 100
    seq_len = 100
    lr = 5e-4
    dropout = 0.3
    batch_size = 196
    heads_count = 4
    embedding_dim = 144
    blocks_count = 4
    max_epochs = 6